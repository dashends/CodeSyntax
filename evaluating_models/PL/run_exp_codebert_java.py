import json
import pickle
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaTokenizerFast



# extract attention and convert attention to word-level

def get_word_word_attention(token_token_attention, words_to_tokens):
    """Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    codebert_tokens_length = token_token_attention.shape[0]
    word_starts = set()
    not_mapped_tokens = [] # store the python tokens that no codebert token is mapped to
    
    # special case: sometimes two python tokens are combined as one in codebert tokens
    # e.g. "):"
    # when we convert to word-level attention, we need to make a copy of the row becase 
    # ")" and ":" are two different python tokens
    # we store such words in conflicting_heads
    conflicting_heads = {}
    for i in range(len(words_to_tokens)):
        word = words_to_tokens[i]
        if len(word) > 0:
            word_starts.add(word[0])
            if i < len(words_to_tokens)-1 and len(words_to_tokens[i+1]) > 0 and word[0] == words_to_tokens[i+1][0]:
                conflicting_heads[i] = None
                not_mapped_tokens.append(i)
            
    not_word_starts = [i for i in range(codebert_tokens_length) if i not in word_starts]
  
    # find python tokens that are mapped to no cubert tokens
    not_mapped_tokens.extend([idx for idx, l in enumerate(words_to_tokens) if l ==[]])
    not_mapped_tokens = sorted(not_mapped_tokens)
    
    # sum up the attentions for all tokens in a word that has been split
    for i, word in enumerate(words_to_tokens):
        if len(word) > 0:
            if i in conflicting_heads:
                conflicting_heads[i] = token_token_attention[:, word].sum(axis=-1)
            else:
                token_token_attention[:, word[0]] = token_token_attention[:, word].sum(axis=-1)
    token_token_attention = np.delete(token_token_attention, not_word_starts, -1)
    # do not delete python token that is not mapped to cubert tokens
    for idx in not_mapped_tokens:
        token_token_attention = np.insert(token_token_attention, idx, 0, axis=-1)
    # resolve the special case that two python tokens are combined as one in codebert tokens
    for i in conflicting_heads.keys():
        token_token_attention[:, i] = conflicting_heads[i]
        
    # combining attention maps for words that have been split
    for i, word in enumerate(words_to_tokens):
        if len(word) > 0:
            if i in conflicting_heads:
                conflicting_heads[i] = np.mean(token_token_attention[word], axis=0)
            else:
                # mean
                token_token_attention[word[0]] = np.mean(token_token_attention[word], axis=0)
#                 # max
#                 token_token_attention[word[0]] = np.max(token_token_attention[word], axis=0)
#                 token_token_attention[word[0]] /= token_token_attention[word[0]].sum()

    token_token_attention = np.delete(token_token_attention, not_word_starts, 0)
    # do not delete python token that is not mapped to cubert tokens
    for idx in not_mapped_tokens:
        token_token_attention = np.insert(token_token_attention, idx, 0, axis=0)
    # resolve the special case that two python tokens are combined as one in codebert tokens
    for i in conflicting_heads.keys():
        token_token_attention[i] = conflicting_heads[i]

    return token_token_attention

def make_attn_word_level(alignment, attn):
    return np.stack([[
        get_word_word_attention(attn_head, alignment)
        for attn_head in layer_attns] for layer_attns in attn])



# run CodeBERT to extract attention and convert attention to word-level
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)
model.eval()

with open("data/CodeBERT_tokenized/java_full.json", 'r') as f:
    examples = json.load(f)


max_attn_data = []
for example in examples:
    if example['id']%100 == 0:
        print(example['id'])
    # if example["id"] in common_ids:
    if len(example['input_ids']) < 512:
        outputs = model(torch.tensor(example['input_ids']).unsqueeze(0), output_attentions=True)
        attn = outputs.attentions # list of tensors of shape 1*12*num_of_tokens*num_of_tokens
        attn = np.vstack([layer.cpu().detach().numpy() for layer in attn]) # shape of 12*12*num_of_tokens*num_of_tokens   12 layers 12 heads
        attn = make_attn_word_level(example["alignment"], attn)
        # example['attns'] = attn

        # preprocess attention by sorting predictions based upon weights
        attn[:, :, range(attn.shape[2]), range(attn.shape[2])] = 0
        max_attn = np.flip(np.argsort(attn, axis=3),axis=3).astype(np.int16)[:,:,:,:20]
        max_attn_data.append({"java_tokens": example["java_tokens"], "tokens": example["tokens"], 
                            "max_attn": max_attn, "id": example["id"]})

        


with open("data/attention/codebert_java_full_attn_sorted.pkl", "wb") as f:
    pickle.dump(max_attn_data,f)



