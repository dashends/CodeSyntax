import pickle

# remove uncommon data points

with open("data/depparse_german/dev_attn_sorted_bert.pkl", "rb") as f:
    bert = pickle.load(f)
with open("data/depparse_german/dev_attn_sorted_xlmr_base.pkl", "rb") as f:
    roberta = pickle.load(f)


bert_ids = set()
roberta_ids = set()
for data in bert:
    bert_ids.add(data["id"])
for data in roberta:
    roberta_ids.add(data["id"])

# uncommon_ids = bert_ids.symmetric_difference(roberta_ids)
# print("we have", len(uncommon_ids), "uncommon datapoints")

for filename, all_data in [("dev_attn_sorted_bert_base_cased", bert),
                            ("dev_attn_sorted_bert_base", None),
                            ("dev_attn_sorted_bert_large_cased", None),
                            ("dev_attn_sorted_bert_large", None),
                         ("dev_attn_sorted_roberta_base", None),
                         ("dev_attn_sorted_roberta_large", roberta)]:
    if all_data == None:
        with open("data/depparse/"+filename+".pkl", "rb") as f:
            all_data = pickle.load(f)
    common_data = []
    for data in all_data:
        if data["id"] not in uncommon_ids:
            common_data.append(data)
    print(filename, "has", len(common_data), "datapoints")
    with open("data/depparse/"+filename+".pkl", "wb") as f:
        pickle.dump(common_data, f)