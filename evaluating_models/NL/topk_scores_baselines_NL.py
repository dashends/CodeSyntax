import json
import pickle
import numpy as np
from collections import defaultdict
# find offset baseline top k score
# similar to attenion heads, we consider each offset predictor that gives k predictions based upon fixed offset
# e.g. an offset head can predict [index+1, index+6, index+2, index+12] for each token for each example
# then we can calculate top k score for this offset predictor.

import itertools



def get_baseline_correctness_NL(max_attn_data, relns, min_offset=-10, max_offset=19):
    print("getting correctness for single baselines.")
    #  Each row is a flatten array of 0/1 that represents correctness for all labels
    # correctness for offset i is stored in row i.
    correctness = {}
    num_predictors = max_offset - min_offset +1
    for reln in relns:
        correctness[reln] = [[] for i in range(num_predictors)]

    for index in range (min_offset, max_offset+1):
        for example in max_attn_data:
            for dep_idx, reln, head_idx in zip(range(len(example["relns"])), example["relns"], example["heads"]):
                prediction = dep_idx + index
                if prediction == head_idx-1:
                    correctness[reln][index].append(1)
                else:
                    correctness[reln][index].append(0)

    return correctness


	
def get_relns_NL(dataset):
    relns = set()
    for example in dataset:
        for reln in example["relns"]:
            relns.add(reln)
    relns = list(relns)
    relns.sort()
    return relns



def get_baseline_topk_scores_NL(correctness, relns, max_k=20, min_offset=-10, max_offset=19):
    # This function selects next best baseline by picking the one that gives highest increse in score 
    print("getting top k scores for each relation")

    num_predictors = max_offset - min_offset +1
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
            single_baseline_scores = [-1]*num_predictors
            for i in range(min_offset, max_offset+1):
                # find the score for the relation
                offset_correctness = reln_correctness[i] & (~topk) # we only care about labels that we have not gotten correct
                score = np.count_nonzero(offset_correctness)/len(offset_correctness)
                single_baseline_scores[i] = score

            # sort baselines
            single_baseline_scores=np.array(single_baseline_scores)
            best_baseline=single_baseline_scores.argsort()[-1]
            best_baseline = -(num_predictors -best_baseline) if best_baseline  > max_offset else best_baseline

            # add the best baseline to the combination
            combination.append(best_baseline)
            topk = topk | reln_correctness[best_baseline]
            score = np.count_nonzero(topk)/len(topk)
            reln_scores_topk[reln][k] = (score, combination.copy())
            
    return reln_scores_topk

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



baseline_type="offset"
max_offset=512
min_offset=-512

# offset
with open("data/depparse_english/dev_attn_sorted_roberta_base.pkl", "rb") as f:
# with open("data/depparse_german/dev_attn_sorted_bert.pkl", "rb") as f:
    max_attn_data = pickle.load(f)

relns = get_relns_NL(max_attn_data)
print(relns)

correctness = get_baseline_correctness_NL(max_attn_data, relns, min_offset=min_offset, max_offset=max_offset)
reln_scores_topk_offset = get_baseline_topk_scores_NL(correctness, relns, max_k=20, min_offset=min_offset, max_offset=max_offset)
scores_topk_offset = get_scores_topk(reln_scores_topk_offset, max_k=20)
print(scores_topk_offset)

with open("data/depparse_english/roberta_base_topk_scores_"+baseline_type+".pkl", "wb") as f:
# with open("data/depparse_german/german_topk_scores_"+baseline_type+".pkl", "wb") as f:
    pickle.dump(reln_scores_topk_offset, f)