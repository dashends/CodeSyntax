import pickle

language="java"


print("loading")

with open("data/attention/codebert_"+language+"_full_attn_sorted_test.pkl", "rb") as f:
    codebert_test = pickle.load(f)
with open("data/attention/codebert_"+language+"_full_attn_sorted_valid.pkl", "rb") as f:
    codebert_valid = pickle.load(f)

codebert_ids = set()
cubert_ids = set()
for dataset in [codebert_test, codebert_valid]:
    for data in dataset:
        codebert_ids.add(data["id"])

with open("data/attention/"+language+"_common_ids.pkl", "wb") as f:
        pickle.dump(codebert_ids, f)

with open("data/attention/cubert_"+language+"_full_attn_sorted_test.pkl", "rb") as f:
    cubert_test = pickle.load(f)
with open("data/attention/cubert_"+language+"_full_attn_sorted_valid.pkl", "rb") as f:
    cubert_valid = pickle.load(f)



for dataset in [cubert_test, cubert_valid]:
    for data in dataset:
        cubert_ids.add(data["id"])

uncommon_ids = codebert_ids.symmetric_difference(cubert_ids)
common_ids = codebert_ids.intersection(cubert_ids)
print("we have", len(uncommon_ids), "uncommon datapoints")

for filename, all_data in [("codebert_"+language+"_full_attn_sorted_test", codebert_test),
                         ("codebert_"+language+"_full_attn_sorted_valid", codebert_valid),
                         ("cubert_"+language+"_full_attn_sorted_test", cubert_test),
                         ("cubert_"+language+"_full_attn_sorted_valid", cubert_valid)]:
    common_data = []
    for data in all_data:
        if data["id"] in common_ids:
            common_data.append(data)
    print(filename, "has", len(common_data), "datapoints")
    with open("data/attention/"+filename+"_common.pkl", "wb") as f:
        pickle.dump(common_data, f)