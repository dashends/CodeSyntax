from cubert import python_tokenizer
from cubert import java_tokenizer
from cubert import code_to_subtokenized_sentences
import re
import io
import javalang
import tokenize
from tensor2tensor.data_generators import text_encoder
import pickle
import json

# Align cubert sub-tokens with python token
# Rewrite Attention-analysis’ tokenize_and_align(). 
# returns a list of lists where each sub-list 
# contains cubert-tokenized tokens for the correponding word.
# each cubert-subtoken is represented by int starting from 0
# e.g. [[0, 1], [2]]

# By default, javalang tokenizer does not produce '\n' tokens while python tokenizer does.
# For Java, if we want to add '\n' into tokens, we need to produce different alignments too.

token_map = {'(': '(_', '[': '[_', "{":"{_", "}": "}_", ')': ')_', ']': ']_',
             '    ':'\\u\\u\\uINDENT\\u\\u\\u ', '        ':'\\u\\u\\uINDENT\\u\\u\\u ', '            ':'\\u\\u\\uINDENT\\u\\u\\u ',
            '                ':'\\u\\u\\uINDENT\\u\\u\\u ', '                    ':'\\u\\u\\uINDENT\\u\\u\\u ',
            '       ':'\\u\\u\\uINDENT\\u\\u\\u ', '\n':"\\u\\u\\uNEWLINE\\u\\u\\u_", '\r\n':"\\u\\u\\uNEWLINE\\u\\u\\u_",
            '\n\t':"\\u\\u\\uNEWLINE\\u\\u\\u_"}
ignore_set = set([''])
new_line = set(['\n', '\r\n', '\n\t'])

def align_cubert_tokens(cubert_tokens, python_tokens, id=-1):
    result = []
    current_idx = 0
    for python_token in python_tokens:
        # convert python token into cubert token
        if python_token in token_map.keys():
            subtokenized_python_token = [[token_map[python_token], 'EOS']]
        else:
            subtokenized_python_token = (
                  code_to_subtokenized_sentences.code_to_cubert_sentences(
                      code=python_token,
                      initial_tokenizer=tokenizer,
                      subword_tokenizer=subword_tokenizer))
#         print(python_token, subtokenized_python_token)
        #ignore invlaid tokens and comments
        if len(subtokenized_python_token) >0 and subtokenized_python_token[0][0]!='#^_':
            subtokenized_python_token = subtokenized_python_token[0][0:-1]

            length = len(subtokenized_python_token)
            remaining_cubert_tokens = cubert_tokens[current_idx:]
            found = False
            
            if python_token in new_line:
#                 print(repr(python_token), remaining_cubert_tokens[0:1])
                if remaining_cubert_tokens[0] == "\\u\\u\\uNEWLINE\\u\\u\\u_" or remaining_cubert_tokens[0] == "\\u\\u\\uNL\\u\\u\\u_":
                    result.append([*range(current_idx, current_idx+1)])
                    current_idx += 1
                    found = True
            else:
                for i in range(len(remaining_cubert_tokens) - length +1):
                    if remaining_cubert_tokens[i:i+length] == subtokenized_python_token and (i == 0 or not remaining_cubert_tokens[i-1].endswith("u^_")):
#                         print('skipping', i,'tokens.', repr(python_token), remaining_cubert_tokens[i:i+length])
    #                     print('remaining tokens', remaining_cubert_tokens[i+length:i+length+5])
                        result.append([*range(current_idx+i, current_idx+i+length)])
                        current_idx += length+i
                        found = True
                        break
            if not found:
                if (not python_token.startswith('"""') and not python_token.startswith("'''") 
                                                and not python_token.startswith("#") and python_token not in new_line):
                    print("id =", id, ": python token not found: ",  repr(python_token))
                    print('remaining tokens', remaining_cubert_tokens[0:10])
                result.append([])
            
        else:
            if python_token not in ignore_set and not python_token.startswith("#") and not python_token.isspace():
                print("ignoring python token: ", repr(python_token)) 
            result.append([])
        
    return result


tokenizer = python_tokenizer.PythonTokenizer()
subword_tokenizer = text_encoder.SubwordTextEncoder("cubert/vocab.txt")



# save subtoken alignment to json
with open('../../generating_CodeSyntax/deduplicated_python_code.pickle', 'rb') as f:
    sample = pickle.load(f)
examples = []

total_count=len(sample)
batch_count=int(total_count/1000)+1
for b in range(0, batch_count):
    examples = []
    for i in range(1000*b, min(1000*(b+1),total_count)):
        try:
            cubert_tokens = (code_to_subtokenized_sentences.code_to_cubert_sentences(
              code=sample[i],
              initial_tokenizer=tokenizer,
              subword_tokenizer=subword_tokenizer))
            tokens = ["[CLS]_"]
            for sentence in cubert_tokens:
                tokens.extend(sentence)
            tokens.append("[SEP]_")
            if len(tokens) < 512:
                # generate alignment
                python_tokens = []
                for token in tokenize.generate_tokens(io.StringIO(sample[i]).readline):
                    python_tokens.append(token.string)
                if tokens == ['[CLS]_', '\\u\\u\\u', 'ERROR', '\\u\\u\\u_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '[SEP]_']:
                    print("skipping", i, "because of cubert tokenization error")
                    continue
                alignment = align_cubert_tokens(tokens, python_tokens, id=i)
                examples.append({"tokens": tokens, "id": i, "alignment": alignment, "python_tokens": python_tokens})
        except Exception as e:
            print("skipping", i, "because of", e)
    print("length of batch", b, ":", len(examples))

    with open("data/CuBERT_tokenized/python_"+str(1000*b)+"_"+str(1000*(b+1))+".json", 'w') as f:
        json.dump(examples, f, indent=2)
        