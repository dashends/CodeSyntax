# This file is obtained from https://github.com/clarkkev/attention-analysis

"""Going from BERT's bpe tokenization to word-level tokenization."""

import utils
from bert import tokenization

import numpy as np


def tokenize_and_align(tokenizer, words, cased):
  """Given already-tokenized text (as a list of strings), returns a list of
  lists where each sub-list contains BERT-tokenized tokens for the
  correponding word."""

  words = ["[CLS]"] + words + ["[SEP]"]
  basic_tokenizer = tokenizer.basic_tokenizer
  tokenized_words = []
  for word in words:
    word = tokenization.convert_to_unicode(word)
    word = basic_tokenizer._clean_text(word)
    if word == "[CLS]" or word == "[SEP]":
      word_toks = [word]
    else:
      if not cased:
        word = word.lower()
        word = basic_tokenizer._run_strip_accents(word)
      word_toks = basic_tokenizer._run_split_on_punc(word)

    tokenized_word = []
    for word_tok in word_toks:
      tokenized_word += tokenizer.wordpiece_tokenizer.tokenize(word_tok)
    tokenized_words.append(tokenized_word)

  i = 0
  word_to_tokens = []
  for word in tokenized_words:
    tokens = []
    for _ in word:
      tokens.append(i)
      i += 1
    word_to_tokens.append(tokens)
  assert len(word_to_tokens) == len(words)

  return word_to_tokens


def get_word_word_attention(token_token_attention, words_to_tokens, length,
                            mode="mean"):
  """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings)."""

  word_starts = set()
  for word in words_to_tokens:
    if len(word) > 0:
      word_starts.add(word[0])
  not_word_starts = [i for i in range(length) if i not in word_starts]
  
  # find python tokens that are mapped to no cubert tokens
  not_mapped_tokens = [idx for idx, l in enumerate(words_to_tokens) if l ==[]]
  
  # sum up the attentions for all tokens in a word that has been split
  for word in words_to_tokens:
    if len(word) > 0:
      token_token_attention[:, word[0]] = token_token_attention[:, word].sum(axis=-1)
  token_token_attention = np.delete(token_token_attention, not_word_starts, -1)
  # do not delete python token that is not mapped to cubert tokens
  for idx in not_mapped_tokens:
    token_token_attention = np.insert(token_token_attention, idx, 0, axis=-1)

  # several options for combining attention maps for words that have been split
  # we use "mean" in the paper
  for word in words_to_tokens:
    if len(word) > 0:
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
  # do not delete python token that is not mapped to cubert tokens
  for idx in not_mapped_tokens:
    token_token_attention = np.insert(token_token_attention, idx, 0, axis=0)

  return token_token_attention


def make_attn_word_level(data):
  for features in utils.logged_loop(data):
    words_to_tokens = features["alignment"]
    length = len(features["tokens"])
    features["attns"] = np.stack([[
        get_word_word_attention(attn_head, words_to_tokens, length)
        for attn_head in layer_attns] for layer_attns in features["attns"]])
    


def make_attn_block_level(data, ast_dataset, k=3, mode="mean"):
  to_remove = []
  for features in utils.logged_loop(data):
    print(features["id"])
    blocks_to_words = ast_dataset[features["id"]]["blocks"]
    if len(blocks_to_words) == 0:
      to_remove.append(features)
    else:
      features["attns"] = np.stack([[
          get_block_block_attention(attn_head, blocks_to_words, k=k, mode=mode)
          for attn_head in layer_attns] for layer_attns in features["attns"]])
  for item in to_remove:
    data.remove(item)

# find top k args along axis 1
# if not enough values exist in each row, then return original array
def argtopk(A, k):
    if k >= A.shape[1]:
      return A
    else:
      top_k = np.argpartition(A, -k)[:, -k:]
      x = A.shape[0]
      return A[np.repeat(np.arange(x), k), top_k.ravel()].reshape(x, k)

def get_block_block_attention(word_word_attention, blocks_to_words, k,
                            mode="mean"):
  """Convert token-token attention to word-word attention (when tokens are
  derived from words using something like byte-pair encodings)."""

  if type(word_word_attention) != np.ndarray:
    word_word_attention = np.array(word_word_attention)
  block_block_attention = np.zeros([word_word_attention.shape[0], len(blocks_to_words)], dtype=np.float16)


  # average the attentions for all words in a block
  if mode == "topk":
    for i, block in enumerate(blocks_to_words):
      block_block_attention[:, i] = np.mean(argtopk(word_word_attention[:, block[0]:(block[1]+1)], k), axis=-1)
  elif mode == "mean":
    for i, block in enumerate(blocks_to_words):
      block_block_attention[:, i] = np.mean(word_word_attention[:, block[0]:(block[1]+1)], axis=-1)
  for i in range(word_word_attention.shape[0]):
      sum = block_block_attention[i].sum()
      if sum != 0:
        block_block_attention[i] /= block_block_attention[i].sum()

  # several options for combining attention maps for words that have been split
  # we use "mean" in the paper
  word_word_attention = block_block_attention
  block_block_attention = np.zeros([len(blocks_to_words), len(blocks_to_words)], dtype=np.float16)
  for i, block in enumerate(blocks_to_words):
      if mode == "mean" or mode == "topk":
        block_block_attention[i] = np.mean(word_word_attention[block[0]:(block[1]+1)], axis=0)
        sum = block_block_attention[i].sum()
        if sum != 0:
          block_block_attention[i] /= block_block_attention[i].sum()
      elif mode == "max":
        block_block_attention[i] = np.max(block_block_attention[block[0]:(block[1]+1)], axis=0)
        sum = block_block_attention[i].sum()
        if sum != 0:
          block_block_attention[i] /= block_block_attention[i].sum()
      else:
        raise ValueError("Unknown aggregation mode", mode)

  
  return block_block_attention