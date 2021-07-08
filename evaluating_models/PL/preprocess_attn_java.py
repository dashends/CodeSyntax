import json
import pickle
import numpy as np


# preprocess attention
# find the token that is predicted to be head for each single attention head
def load_pickle(fname):
  with open(fname, "rb") as f:
    return pickle.load(f)  
    

for i in range(0, 14):
    max_attn_data = []
    print("processing java_" + str(i*1000)+"_" + str((i+1)*1000) + "_attn.pkl")
    attn_data = load_pickle("data/CuBERT_tokenized/java_" + str(i*1000)+"_" + str((i+1)*1000) + "_attn.pkl")
    for data in attn_data:
        # if data["id"] in common_ids:
            # cls and sep are already removed from word-level attention
            attn = data["attns"]
            attn[:, :, range(attn.shape[2]), range(attn.shape[2])] = 0
            max_attn = np.flip(np.argsort(attn, axis=3),axis=3).astype(np.int16)[:,:,:,:20]
            max_attn_data.append({"tokens": data["tokens"], "max_attn": max_attn, "id": data["id"]})
    
    with open("data/attention/cubert_java_" + str(i*1000)+"_" + str((i+1)*1000) + "_attn_sorted.pkl", "wb") as f:
        pickle.dump(max_attn_data,f)
    
    del max_attn_data
    del attn_data


# merge into one file
max_attn_data = []
for i in range(0, 14):
    print("loading java_" + str(i*1000)+"_" + str((i+1)*1000) + "_attn_sorted.pkl")
    with open("data/attention/cubert_java_" + str(i*1000)+"_" + str((i+1)*1000) + "_attn_sorted.pkl", "rb") as f:
        max_attn_data.extend(pickle.load(f))

with open("data/attention/cubert_java_full_attn_sorted_valid.pkl", "wb") as f:
    pickle.dump([c for c in max_attn_data if c["id"]>=11009], f)

with open("data/attention/cubert_java_full_attn_sorted_test.pkl", "wb") as f:
    pickle.dump([c for c in max_attn_data if c["id"]<11009], f)