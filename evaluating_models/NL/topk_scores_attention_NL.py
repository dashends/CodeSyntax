import json
import pickle
import numpy as np
from collections import defaultdict



def evaluate_single_head_topk_NL(max_attn_data, relns, attn_layer, attn_head, max_k = 20):
    scores = {}
    n_correct = {}
    n_incorrect = {}
    for reln in relns:
        scores[reln] = np.zeros((max_k+1), dtype = 'float')
        n_correct[reln] = np.zeros((max_k+1), dtype = 'int')
        n_incorrect[reln] = np.zeros((max_k+1), dtype = 'int')

    for example in max_attn_data:
        for dep_idx, reln, head_idx in zip(range(len(example["relns"])), example["relns"], example["heads"]):
            if reln in relns:
                for k in range(1, max_k+1):
                    if k-1 < len(example["max_attn"][attn_layer][attn_head][dep_idx]):
                        k_th_prediction = example["max_attn"][attn_layer][attn_head][dep_idx][k-1]
                        if k_th_prediction == head_idx - 1:
                            n_correct[reln][k:] = [c+1 for c in n_correct[reln][k:]]
                            break
                        else:
                            n_incorrect[reln][k] += 1
                    else:
                        n_incorrect[reln][k:] += 1
                        break

    for reln in relns:
        for k in range(1, max_k+1):
            if (n_correct[reln][k] + n_incorrect[reln][k]) == 0:
                scores[reln][k] = None
            else:
                scores[reln][k] = n_correct[reln][k] / float(n_correct[reln][k] + n_incorrect[reln][k])
    return scores
    
def get_relns_NL(dataset):
    relns = set()
    for example in dataset:
        for reln in example["relns"]:
            relns.add(reln)
    relns = list(relns)
    relns.sort()
    return relns

# scores[reln][layer][head]
def get_scores_NL(max_attn_data, relns, max_k=20):
    scores = {}
    n_correct = {}
    n_total = {}
    num_layer = max_attn_data[0]["max_attn"].shape[0]
    num_head = max_attn_data[0]["max_attn"].shape[1]
    for reln in relns:
        scores[reln] = np.zeros((num_layer, num_head, max_k+1), dtype = 'float')
        n_correct[reln] = np.zeros((num_layer, num_head, max_k+1), dtype = 'int')
        n_total[reln] = 0
        

    for i, example in enumerate(max_attn_data):
        if i % 1000 == 0:
            print("processing example", i)
        n_words = example["max_attn"].shape[3]
        for dep_idx, reln, head_idx in zip(range(len(example["relns"])), example["relns"], example["heads"]):
            n_total[reln] += 1
            for k in range(1, min(max_k+1, n_words+1)):
                k_th_prediction = example["max_attn"][:,:, dep_idx, k-1]
                n_correct[reln][:,:,k:][np.where(k_th_prediction == head_idx - 1)] += 1

    for reln in relns:
        if (n_total[reln]) == 0:
                scores[reln][:,:,:] = -100000
        else:
            for k in range(1, max_k+1):
                scores[reln][:,:,k] = n_correct[reln][:,:,k] / n_total[reln]

    return scores





# average topk scores for each relationship and categories (word level)
def get_avg_NL(scores, relns, max_k=20):
    reln_avg = [None]*(max_k+1)

    for k in range(1, (max_k+1)):
        sum, count = 0, 0
        for reln in relns:
            flatten_idx = np.argmax(scores[reln][:,:,k])
            num_head = scores[reln].shape[1]
            # print(num_head)
            row = int(flatten_idx/num_head)
            col = flatten_idx % num_head
            sum += scores[reln][row][col][k]
            count += 1
        reln_avg[k] = sum/count
    return reln_avg

def print_attn_table_NL(k, relns, scores):
    print("relationship\t\t  accuracy\tlayer\thead")
    sum, count = 0, 0
    table = ""
    for reln in relns:
        flatten_idx = np.argmax(scores[reln][:,:,k])
        num_head = scores[reln].shape[1]
        row = int(flatten_idx/num_head)
        col = flatten_idx % num_head
        table += reln.ljust(30) + str(round(scores[reln][row][col][k],3)).ljust(5) + "\t" + str(row) + "\t" + str(col) + '\n'
        sum += scores[reln][row][col][k]
        count += 1
    print(table)
    print("average of",count,"relations:", sum/count)
    


# with open("data/depparse_english/dev_attn_sorted_bert_large.pkl", "rb") as f:
# with open("data/depparse_german/dev_attn_sorted_xlmr_base.pkl", "rb") as f:
with open("data/depparse_english/dev_attn_sorted_codebert.pkl", "rb") as f:
    max_attn_data = pickle.load(f)
relns = get_relns_NL(max_attn_data)
# relns = ["pobj"]
print("relations", relns)



scores = get_scores_NL(max_attn_data, relns, max_k=20)
reln_avg = get_avg_NL(scores, relns, max_k=20)

print_attn_table_NL(1, relns, scores)

# with open("data/depparse_english/bert_large_topk_scores.pkl", "wb") as f:
# with open("data/depparse_english/roberta_large_topk_scores.pkl", "wb") as f:
# with open("data/depparse_german/german_xlmr_base_topk_scores.pkl", "wb") as f:
with open("data/depparse_english/NL_codebert_topk_scores.pkl", "wb") as f:
    pickle.dump(scores, f)