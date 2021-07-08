import pickle
import numpy as np


# preprocess attention
# find the token that is predicted to be head for each single attention head
def load_pickle(fname):
  with open(fname, "rb") as f:
    return pickle.load(f)  
    

max_attn_data = []
# attn_data = load_pickle("data/depparse/dev_attn_bert_base.pkl")
attn_data = load_pickle("data/depparse_german/dev_attn.pkl")
for i, data in enumerate(attn_data):
    if i % 1000 == 0:
        print("processing example", i)
    # cls and sep have not been removed from word-level attention
    attn = data["attns"]
    attn[:, :, range(attn.shape[2]), range(attn.shape[2])] = 0
    attn = attn[:,:, 1:-1, 1:-1]
    max_attn = np.flip(np.argsort(attn, axis=3),axis=3).astype(np.int16)[:,:,:,:20]
    max_attn_data.append({"words": data["words"], "max_attn": max_attn, "relns": data["relns"], "heads": data["heads"], "id": i})

# with open("data/depparse/dev_attn_sorted_bert_base.pkl", "wb") as f:
with open("data/depparse_german/dev_attn_sorted_bert.pkl", "wb") as f:
    pickle.dump(max_attn_data,f)

with open("data/depparse_german/dev_attn_sorted_bert_short.pkl", "wb") as f:
    pickle.dump(max_attn_data[0:1000],f)