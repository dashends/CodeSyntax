import json
import pickle
import numpy as np
from collections import defaultdict
# find offset baseline top k score using dynamic programming
# similar to attenion heads, we consider each offset predictor that gives k predictions based upon fixed offset
# e.g. an offset head can predict [index+1, index+6, index+2, index+12] for each token for each example
# then we can calculate top 1~4 score for this offset predictor.
# approach:
# for each k, enumerate all combinations of offsets of length k, find the score
# + dynamic programming (keep the correctness of length k, then for k+1 only need to do one more binary or)

import itertools


def find_next_keyword(example, start_index, keyword):
    tokens = dataset[example['id']]['tokens']
    for i in range(start_index, len(tokens)):
        if tokens[i] == keyword:
            return i
    return -1

def get_baseline_correctness(max_attn_data, relns, dataset, metric="first", max_offset=19, keywords= []):
    print("getting correctness for single baselines.")
    #  Each row is a flatten array of 0/1 that represents correctness for all labels
    # correctness for offset i is stored in row i-1.
    # correctness for keywords[i] is stored in row max_offset+i
    correctness = {}
    unreachable_percentage=[]
    num_predictors = max_offset + len(keywords)
    for reln in relns:
        correctness[reln] = [[] for i in range(num_predictors)]
        for index in range (1, num_predictors+1):
            for i in range(len(max_attn_data)):
                id = max_attn_data[i]['id']
                if reln in dataset[id]["relns"]:
                    labels = dataset[id]["relns"][reln]
                    for head_idx, dep_range_start_idx, dep_range_end_idx in labels:
                        if index <= max_offset:
                            # current index is offset
                            prediction = head_idx + index
                        else:
                            # current index is keyword
                            prediction = find_next_keyword(max_attn_data[i], head_idx+1, keywords[index-max_offset-1])
                        if ((metric == "first" and prediction == dep_range_start_idx) or
                                (metric == "last" and prediction == dep_range_end_idx) or
                                (metric == "any" and prediction >= dep_range_start_idx and 
                                 prediction <= dep_range_end_idx)):
                            correctness[reln][index-1].append(1)
                        else:
                            correctness[reln][index-1].append(0)
                            
        # find percentage of tokens that are not reachable by offset baselines
        total, out = 0, 0
        for i in range(len(max_attn_data)):
            id = max_attn_data[i]['id']
            if reln in dataset[id]["relns"]:
                labels = dataset[id]["relns"][reln]
                for head_idx, dep_range_start_idx, dep_range_end_idx in labels:
                    total += 1
                    if ((metric == "first" and dep_range_start_idx-head_idx > max_offset) or
                        (metric == "last" and dep_range_end_idx-head_idx > max_offset) or
                        (metric == "any" and dep_range_start_idx-head_idx > max_offset)):
                        out+=1
        unreachable_percentage.append(out/total)
        
    return correctness, unreachable_percentage


def get_baseline_topk_scores(correctness, relns, max_k=20, max_offset=19, keywords= []):
                      
    num_predictors = max_offset + len(keywords)
    reln_scores_topk = {}
    for reln in relns:
        reln_scores_topk[reln] = [(0, [0])]*(max_k+1)

    counter = 0

    # enumerate all combinations of offsets of length k
    for reln in relns:
        topk_lookup = {}
        for k in range (1, (max_k+1)):
            print("calculating top", k, "score for", reln)
            new_topk_lookup = {}
            counter_to_add = int(min(k, int(num_predictors/2))/2)
            reln_correctness = np.array(correctness[reln], dtype=bool)
            for combination in itertools.combinations(range(1, num_predictors+1), k):
                # find the score using dynamic programming
                found = False
                for skip_i in range(k):
                    k_minus_1 = list(combination[0:skip_i])
                    k_minus_1.extend(combination[skip_i+1:])
                    k_minus_1 = tuple(k_minus_1)
                    if k_minus_1 in topk_lookup:
                        topk = topk_lookup[k_minus_1] | reln_correctness[combination[skip_i]-1]
                        found = True
                        break
                if not found:
    #                 if len(combination)>1:
    #                     print("combination not exist", combination)
                    topk = np.zeros((reln_correctness.shape[1]), dtype=bool)
                    for i in combination:
                        topk = topk | reln_correctness[i-1]
                if counter >= counter_to_add:
                    counter = 0
                    new_topk_lookup[combination] = topk
                counter += 1
                score = np.count_nonzero(topk)/len(topk)
                # keep only max score for k
                if reln_scores_topk[reln][k][0] < score:
                    reln_scores_topk[reln][k] = (score, combination)
            # drop dp lookup map for k-1
            topk_lookup = new_topk_lookup
            
            # convert index to keywords
            converted_combination = []
            for index in reln_scores_topk[reln][k][1]:
                converted_combination.append(keywords[index-max_offset-1] if index > max_offset else index)
            reln_scores_topk[reln][k] = (reln_scores_topk[reln][k][0], converted_combination)
                    
    return reln_scores_topk

def get_baseline_mean_rank(reln_scores_topk, correctness, relns, max_k=20, max_offset=19, keywords= []):
    mean_ranks = {}
    for reln in relns:
        topk = []
        for k in range (1, (max_k+1)):
            score, combination = reln_scores_topk[reln][k]
            if combination == 0 or combination ==[0]:
                continue
            for index in combination:
                if index not in topk:
                    topk.append(index)
            assert len(topk) == len(combination)
        rank_sum = 0
        count = len(correctness[reln][1])
        for i in range(0, count):
            correct = False
            for index, prediction in enumerate(topk):
                if isinstance(prediction, str):
                    # convert keyword to index
                    prediction = keywords.index(prediction) + max_offset + 1
                if correctness[reln][prediction-1][i] == 1:
                    rank_sum += index+1
                    correct = True
                    break
            if not correct:
                rank_sum += max_k+1
        mean_ranks[reln] = rank_sum/count, topk
    return mean_ranks




def get_scores_topk(reln_scores_topk, max_k=20):
    scores_topk = [0]*(max_k+1)
    for k in range(1, max_k+1):
        sum, count = 0, 0
        for key, value in reln_scores_topk.items():
            sum += value[k][0]
            count += 1
        scores_topk[k] = sum/count
    last = 0
    for i in range(0, max_k+1):
        if scores_topk[i] == 0:
            scores_topk[i] = last
        else:
            last = scores_topk[i]
    return scores_topk

def get_scores_topk_max_of_two(reln_scores_topk_offset, reln_scores_topk_keyword, max_k=20):
    # replace 0 with the previous score
    for scores in [reln_scores_topk_offset, reln_scores_topk_keyword]:
        for key, value in scores.items():
            last = (0, [0])
            for i in range(0, max_k+1):
                if value[i][0] == 0:
                    value[i] = last
                last = value[i]
    
    scores_topk = [0]*(max_k+1)
    for k in range(1, max_k+1):
        sum, count = 0, 0
        for item1, item2  in zip(reln_scores_topk_offset.items(), reln_scores_topk_keyword.items()):
            key1, value1 = item1
            key2, value2 = item2
            sum += max(value1[k][0], value2[k][0])
            count += 1
        scores_topk[k] = sum/count

    last = 0
    for i in range(0, max_k+1):
        if scores_topk[i] == 0:
            scores_topk[i] = last
        else:
            last = scores_topk[i]
    return scores_topk

	
def get_relns(dataset):
    relns = set()
    for example in dataset:
        for reln in example["relns"].keys():
            relns.add(reln)
    relns = list(relns)
    relns.sort()
    return relns


def get_baseline_topk_scores(correctness, relns, max_k=20, max_offset=19, keywords= []):
    # # This function selects next best baseline by picking the one that gives highest increse in score 
    num_predictors = max_offset + len(keywords)
    reln_scores_topk = {} # reln -> list of top k scores (index is k)
    for reln in relns:
        reln_scores_topk[reln] = [(0, [0])]*(max_k+1)

            
   
    for reln in relns:
        reln_correctness = np.array(correctness[reln], dtype=bool)
        topk = np.zeros((reln_correctness.shape[1]), dtype=bool)
        combination=[]
        # selects next baseline by picking the one that gives highest increse in score 
        for k in range (1, (max_k+1)):
            # calculate single baseline score
            single_baseline_scores = []
            for i in range(num_predictors):
                # find the score for the relation
                offset_correctness = reln_correctness[i] & (~topk) # we only care about labels that we have not gotten correct
                score = np.count_nonzero(offset_correctness)/len(offset_correctness)
                single_baseline_scores.append(score)   

            # sort baselines
            single_baseline_scores=np.array(single_baseline_scores)
            best_baseline=single_baseline_scores.argsort()[-1]
    
            # add the best baseline to the combination
            combination.append(best_baseline)
            topk = topk | reln_correctness[best_baseline]
            score = np.count_nonzero(topk)/len(topk)
            reln_scores_topk[reln][k] = (score, combination)
            
            # convert index to keywords
            converted_combination = []
            for index in reln_scores_topk[reln][k][1]:
                index = index+1
                converted_combination.append(keywords[index-max_offset-1] if index > max_offset else index)
            reln_scores_topk[reln][k] = (reln_scores_topk[reln][k][0], converted_combination)
                    
    return reln_scores_topk


with open("../../CodeSyntax/CodeSyntax_python.json", "r") as f:
    dataset = json.load(f)


baseline_type="offset" # offset, keywords, offset_keywords
max_offset=512
keywords = ['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
baseline_types =  ["offset", "keywords", "offset_keywords"]
type_to_offsets = {"offset": (max_offset, []), "keywords": (0, keywords), "offset_keywords": (max_offset, keywords)}
for metric in ["first", "any", "last"]:
    for partition in ["valid", "test"]:
        # offset
        baseline_type = baseline_types[0]
        max_offset, keywords = type_to_offsets[baseline_type]
        with open("data/attention/cubert_python_full_attn_sorted_"+partition+"_common.pkl", "rb") as f:
            max_attn_data = pickle.load(f)

        relns = get_relns(dataset)
        correctness, unreachable_percentage = get_baseline_correctness(max_attn_data, relns, dataset, metric=metric, max_offset=max_offset, keywords=keywords)
        reln_scores_topk_offset = get_baseline_topk_scores(correctness, relns, max_k=20, max_offset=max_offset, keywords=keywords)
        scores_topk_offset = get_scores_topk(reln_scores_topk_offset, max_k=20)
        print(scores_topk_offset)

        with open("data/scores/python_full_topk_scores_"+baseline_type+"_"+partition+"_"+metric+"_common.pkl", "wb") as f:
            pickle.dump(reln_scores_topk_offset, f)

        # keywords
        baseline_type = baseline_types[1]
        max_offset, keywords = type_to_offsets[baseline_type]
        correctness, unreachable_percentage = get_baseline_correctness(max_attn_data, relns, dataset, metric=metric, max_offset=max_offset, keywords=keywords)
        reln_scores_topk_keywords = get_baseline_topk_scores(correctness, relns, max_k=20, max_offset=max_offset, keywords=keywords)
        scores_topk_keywords = get_scores_topk(reln_scores_topk_keywords, max_k=20)
        print(scores_topk_keywords)

        with open("data/scores/python_full_topk_scores_"+baseline_type+"_"+partition+"_"+metric+"_common.pkl", "wb") as f:
            pickle.dump(reln_scores_topk_keywords, f)

        # max(offset, keywords) for each relation
        baseline_type = baseline_types[2]
        scores_topk_offset_keywords = get_scores_topk_max_of_two(reln_scores_topk_offset, reln_scores_topk_keywords, max_k=20)
        print(scores_topk_offset_keywords)

        # with open("data/scores/python_full_topk_scores_"+baseline_type+"_"+partition+"_"+metric+".pkl", "wb") as f:
        #     pickle.dump(scores_topk_offset_keywords, f)