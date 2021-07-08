import pickle

max_attn_data = []
for k in range(6):
    with open("data/depparse/dev_attn_sorted_roberta_large"+str(k*10000)+"_"+str((k+1)*10000)+".pkl", "rb") as f:
        max_attn_data.extend(pickle.load(f))

with open("data/depparse/dev_attn_sorted_roberta_large.pkl", "wb") as f:
    pickle.dump(max_attn_data,f)