import json
import pickle
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaTokenizerFast, XLMRobertaTokenizerFast, XLMRobertaModel



# extract attention and convert attention to word-level

def align_codebert_tokens(codebert_encodings, words, id=-1):
    start, end = 0, 0
    result = []
    # produce alignment by using start and end
    for word in words:
        end = start+len(word)
        codebert_token_index_first = codebert_encodings.char_to_token(start)
        codebert_token_index_last = codebert_encodings.char_to_token(end-1)
        result.append([*range(codebert_token_index_first, codebert_token_index_last+1)])
#         print(repr(word), codebert_encodings.tokens()[codebert_token_index_first: codebert_token_index_last+1])  
        start = end + 1
    assert len(result) == len(words) # assert that every word is mapped to some codebert tokens
    for tokens in result:
        assert len(tokens) > 0
    tokens = [item for sublist in result for item in sublist]
    assert len(tokens) == len(set(tokens)) # assert that  no codebert token is mapped twice
    return result

def get_word_word_attention(token_token_attention, words_to_tokens, length,
                            mode="mean"):
    """This function is adopted from paper "What Does BERT Look At? An Analysis of BERT's Attention"
    Convert token-token attention to word-word attention (when tokens are
    derived from words using something like byte-pair encodings)."""

    word_starts = set()
    for word in words_to_tokens:
        word_starts.add(word[0])
    not_word_starts = [i for i in range(length) if i not in word_starts]
  
    # sum up the attentions for all tokens in a word that has been split
    for word in words_to_tokens:
        token_token_attention[:, word[0]] = token_token_attention[:, word].sum(axis=-1)
    token_token_attention = np.delete(token_token_attention, not_word_starts, -1)

  # several options for combining attention maps for words that have been split
  # we use "mean" in the paper
    for word in words_to_tokens:
        if mode == "first":
            pass
        elif mode == "mean":
            token_token_attention[word[0]] = np.mean(token_token_attention[word], axis=0)
        elif mode == "max":
            token_token_attention[word[0]] = np.max(token_token_attention[word], axis=0)
            token_token_attention[word[0]] /= token_token_attention[word[0]].sum()
        else:
            raise ValueError("Unknown aggregation mode", mode)
    token_token_attention = np.delete(token_token_attention, not_word_starts, 0)


    return token_token_attention


def make_attn_word_level(alignment, attn, length):
    """This function is adopted from paper What Does BERT Look At? An Analysis of BERT's Attention"""
    return np.stack([[
        get_word_word_attention(attn_head, alignment, length)
        for attn_head in layer_attns] for layer_attns in attn])



# run CodeBERT to extract attention and convert attention to word-level
config = "roberta-large" # "xlm-roberta-large" "roberta-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = RobertaTokenizerFast.from_pretrained(config)
# model = RobertaModel.from_pretrained(config)
tokenizer = XLMRobertaTokenizerFast.from_pretrained(config)
model = XLMRobertaModel.from_pretrained(config)
model.to(device)
model.eval()

with open("data/depparse_german/dev.json", 'r') as f:
# with open("data/depparse_english/dev.json", 'r') as f:
    examples = json.load(f)

max_attn_data = []
for i, example in enumerate(examples):
    if i%100 == 0:
        print(i)
    words = example["words"]
    encodings = tokenizer(" ".join(words))
    if len(encodings.tokens()) < 512:
        input_tensor = torch.tensor(encodings['input_ids'], device=device).unsqueeze(0)
        outputs = model(input_tensor, output_attentions=True)
        attn = outputs.attentions # list of tensors of shape 1*12*num_of_tokens*num_of_tokens
        attn = np.vstack([layer.cpu().detach().numpy() for layer in attn]) # shape of 12*12*num_of_tokens*num_of_tokens   12 layers 12 heads
        attn = make_attn_word_level(align_codebert_tokens(encodings, words, id = i), attn, len(encodings.tokens()))
        # example['attns'] = attn

        # preprocess attention by sorting predictions based upon weights
        attn[:, :, range(attn.shape[2]), range(attn.shape[2])] = 0
        max_attn = np.flip(np.argsort(attn, axis=3),axis=3).astype(np.int16)[:,:,:,:20]
        max_attn_data.append({"words": words, "max_attn": max_attn, "relns": example["relns"], "heads": example["heads"], "id": i})

        


# with open("data/depparse_english/dev_attn_sorted_roberta_base.pkl", "wb") as f:
with open("data/depparse_german/dev_attn_sorted_xlmr_base.pkl", "wb") as f:
    pickle.dump(max_attn_data,f)



