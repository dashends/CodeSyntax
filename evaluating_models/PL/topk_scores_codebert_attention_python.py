import json
import pickle
import numpy as np
from collections import defaultdict
import itertools

# Code for evaluating individual attention maps and baselines
# at word level
# metric: "first", "any", "last"
# Code for evaluating individual attention maps and baselines
# at word level
# metric: "first", "any", "last"
def evaluate_predictor_topk(max_attn_data, dataset, reln, attn_layer, attn_head, max_k = 20, metric="first"):
    scores = np.zeros((max_k+1), dtype = 'float')
    n_correct = [0]*(max_k+1)
    n_incorrect = [0]*(max_k+1)
    for index, example in enumerate(max_attn_data):
        i = example['id']
        if reln in dataset[i]["relns"]:
            for head_idx, dep_range_start_idx, dep_range_end_idx in dataset[i]["relns"][reln]:
                for k in range(1, max_k+1):
                    k_th_prediction = example["max_attn"][attn_layer][attn_head][head_idx][k-1]
                    if ((metric == "first" and k_th_prediction == dep_range_start_idx) or
                                (metric == "last" and k_th_prediction == dep_range_end_idx) or
                                (metric == "any" and k_th_prediction >= dep_range_start_idx and k_th_prediction <= dep_range_end_idx)):
                        n_correct[k:] = [c+1 for c in n_correct[k:]]
                        break
                    else:
                        n_incorrect[k] += 1
    for k in range(1, max_k+1):
        if (n_correct[k] + n_incorrect[k]) == 0:
            scores[k] = None
        else:
            scores[k] = n_correct[k] / float(n_correct[k] + n_incorrect[k])
    return scores


    
    
def get_relns(dataset):
    relns = set()
    for example in dataset:
        for reln in example["relns"].keys():
            relns.add(reln)
    relns = list(relns)
    relns.sort()
    return relns

# scores[reln][layer][head]
def get_scores(max_attn_data, dataset, relns, max_k=20, metric="first"):
    scores = {}
    for reln in relns:
        print("processing relationship: ", reln)
        num_layer = max_attn_data[0]["max_attn"].shape[0]
        num_head = max_attn_data[0]["max_attn"].shape[1]
        scores[reln] = np.zeros((num_layer, num_head, max_k+1), dtype = 'float')
        for layer in range(num_layer):
            for head in range(num_head):
#                 if head == 0:
#                     print("layer: ", layer)
                scores[reln][layer][head] = evaluate_predictor_topk(max_attn_data, dataset, reln, layer, head, max_k, metric)
    return scores

categories = {'Control Flow': ['If', 'For', 'While', 'Try'],
        'Expressions': ['BinOp', 'BoolOp', 'Compare', 'Call', 'IfExp', 'Attribute'], 
        'Expr-Subscripting': ['Subscript'],
        'Statements': ['Assign', 'AugAssign'],
        'Vague': ['children']
        }


# average topk scores for each relationship and categories (word level)
def get_avg(scores, relns, max_k=20):
    reln_avg = [None]*(max_k+1)
    cat_avg = {}
    for cat, cat_relns in categories.items():
        cat_avg[cat] = [None]*(max_k+1)

    for k in range(1, (max_k+1)):
        sum, count = 0, 0
        for cat, cat_relns in categories.items():
            cat_sum, cat_count = 0, 0
            for cat_reln in cat_relns:
                for reln in relns:
                    if reln.startswith(cat_reln+":"):
                        flatten_idx = np.argmax(scores[reln][:,:,k])
                        num_head = scores[reln].shape[1]
                        row = int(flatten_idx/num_head)
                        col = flatten_idx % num_head
                        sum += scores[reln][row][col][k]
                        count += 1
                        cat_sum += scores[reln][row][col][k]
                        cat_count += 1
            cat_avg[cat][k] = cat_sum/cat_count
        reln_avg[k] = sum/count
    return (reln_avg, cat_avg)


def print_attn_table(k, relns, scores):
    print("relationship\t\t  accuracy\tlayer\thead")
    sum, count = 0, 0
    table = ""
    table2 = "category\t\t  average accuracy\n"
    for cat, cate_relns in categories.items():
        table += "===================" + cat.ljust(20,"=") + "==========\n"
        cate_sum, cate_count = 0, 0
        for cate_reln in cate_relns:
            for reln in relns:
                if reln.startswith(cate_reln+":"):
                    flatten_idx = np.argmax(scores[reln][:,:,k])
                    num_head = scores[reln].shape[1]
                    row = int(flatten_idx/num_head)
                    col = flatten_idx % num_head
                    table += reln.ljust(30) + str(round(scores[reln][row][col][k],3)).ljust(5) + "\t" + str(row) + "\t" + str(col) + '\n'
                    sum += scores[reln][row][col][k]
                    count += 1
                    cate_sum += scores[reln][row][col][k]
                    cate_count += 1
        table2 += cat.ljust(20) + "\t\t"+str(round(cate_sum/cate_count,3)) + "\n"
    print(table)
    print(table2)
    print("average of",count,"relations:", sum/count)

    
    
def print_baseline_table(k, relns, reln_scores_topk):
    print("relationship\t\t  accuracy\toffset")
    sum, count = 0, 0
    table = ""
    table2 = "category\t\t  average accuracy\n"
    for cat, cate_relns in categories.items():
        table += "===================" + cat.ljust(20,"=") + "==========\n"
        cate_sum, cate_count = 0, 0
        for cate_reln in cate_relns:
            for reln in relns:
                if reln.startswith(cate_reln+":"):
                    table += reln.ljust(30) + str(round(reln_scores_topk[reln][k][0],3)).ljust(5) + "\t" + str(reln_scores_topk[reln][k][1])[1:-1] + '\n'
                    sum += reln_scores_topk[reln][k][0]
                    count += 1
                    cate_sum += reln_scores_topk[reln][k][0]
                    cate_count += 1
        table2 += cat.ljust(20) + "\t\t"+str(round(cate_sum/cate_count,3)) + "\n"
    print(table)
    print(table2)
    print("average of",count,"relations:", sum/count)
    
def print_attn_baseline_table(k, relns, attn_scores, baseline_reln_scores_topk):
    print("relationship\t\tattention\tbaseline\toffset")
    attn_sum, count, baseline_sum = 0, 0, 0
    table = ""
    table2 = "category\t\tattention\tbaseline\n"
    for cat, cate_relns in categories.items():
        table += "=========================" + cat.ljust(20,"=") + "================\n"
        attn_cate_sum, cate_count, baseline_cate_sum = 0, 0, 0
        for cate_reln in cate_relns:
            for reln in relns:
                if reln.startswith(cate_reln+":"):
                    flatten_idx = np.argmax(attn_scores[reln][:,:,k])
                    num_head = attn_scores[reln].shape[1]
                    row = int(flatten_idx/num_head)
                    col = flatten_idx % num_head
                    table += reln.ljust(30) + str(round(attn_scores[reln][row][col][k],3)).ljust(5) + "\t" + str(round(baseline_reln_scores_topk[reln][k][0],3)).ljust(5) + "\t\t" + str(baseline_reln_scores_topk[reln][k][1])[1:-1] + '\n'
                    attn_sum += attn_scores[reln][row][col][k]
                    baseline_sum += baseline_reln_scores_topk[reln][k][0]
                    count += 1
                    attn_cate_sum += attn_scores[reln][row][col][k]
                    baseline_cate_sum += baseline_reln_scores_topk[reln][k][0]
                    cate_count += 1
        table2 += cat.ljust(20) + "\t\t"+str(round(attn_cate_sum/cate_count,3)) + "\t"+str(round(baseline_cate_sum/cate_count,3)) + "\n"
    print(table)
    print(table2)
    print("attention average of",count,"relations:", attn_sum/count)
    print("baseline average of",count,"relations:", baseline_sum/count)

    
    



# max_attn_data = []
# with open("data/attention/codebert_python_full_attn_sorted.pkl", "rb") as f:
#     max_attn_data = pickle.load(f)
# with open("data/attention/codebert_python_full_attn_sorted_valid.pkl", "wb") as f:
#     pickle.dump([c for c in max_attn_data if c["id"]>=8680], f)
# with open("data/attention/codebert_python_full_attn_sorted_test.pkl", "wb") as f:
#     pickle.dump([c for c in max_attn_data if c["id"]<8680], f)



add_new_lines=False
dataset_file_name = "CodeSyntax_python"
output_file_name = "" if not add_new_lines else  "_with_new_lines"
if add_new_lines:
    dataset_file_name += "_with_new_lines"
with open("../../CodeSyntax/"+dataset_file_name+".json", 'r') as f:
    dataset = json.load(f)
relns = get_relns(dataset)
print("relations", relns)

for metric in ["first", "any", "last"]:
    for partition in ["valid", "test"]:
        with open("data/attention/codebert_python_full_attn_sorted_"+partition+"_common.pkl", "rb") as f:
            max_attn_data = pickle.load(f)

        print(partition, metric)
        scores = get_scores(max_attn_data, dataset, relns, max_k=20, metric=metric)
        reln_avg, cat_avg = get_avg(scores, relns, max_k=20)

        print_attn_table(1, relns, scores)

        with open("data/scores/codebert_python_full_topk_scores_"+partition+"_"+metric+output_file_name+"_common.pkl", "wb") as f:
            pickle.dump(scores, f)