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

# Align cubert sub-tokens with java token
# Rewrite Attention-analysis’ tokenize_and_align(). 
# returns a list of lists where each sub-list 
# contains cubert-tokenized tokens for the correponding word.
# each cubert-subtoken is represented by int starting from 0
# e.g. [[0, 1], [2]]

# By default, javalang tokenizer does not produce '\n' tokens while python tokenizer does.
# For Java, if we want to add '\n' into tokens, we need to produce different alignments too.

token_map = {'0': '0_', '\n':"\\u\\u\\uNEWLINE\\u\\u\\u_", '\r\n':"\\u\\u\\uNEWLINE\\u\\u\\u_",	
            '\n\t':"\\u\\u\\uNEWLINE\\u\\u\\u_"}
ignore_set = set([])
new_line = set(['\n', '\r\n', '\n\t'])

def convert_line_offset_to_char_offset_java(line_offset, line_to_length):
    line, offset = line_offset
    offset -= 1
    if line > 1:
        offset += sum(line_to_length[:line-1]) + line-1
    return offset

def convert_char_offset_to_line_offset_java(char_offset, line_to_length):
    line = 0
    while char_offset > line_to_length[line]:
        char_offset -= line_to_length[line]+1
        line += 1  
    return javalang.tokenizer.Position(line+1, char_offset+1)

def tokenize_java_code(code, keep_string_only=False, add_new_lines=False):
    tokens = [token for token in javalang.tokenizer.tokenize(code)]

    if add_new_lines:
        line_to_length = [len(l) for l in code.split("\n")] # stores the number of chars in each line of source code
        token_range = [(convert_line_offset_to_char_offset_java(token.position, line_to_length), 
                        convert_line_offset_to_char_offset_java((token.position[0], token.position[1]+len(token.value)), line_to_length), 
                        token.value) for token in tokens]

        i = 0
        while i < len(tokens) -1:
            code_slice = code[token_range[i][1]:token_range[i+1][0]]
            matches = re.finditer("\n", code_slice)
            new_line_index=[match.start()+token_range[i][1] for match in matches]
            tokens = tokens[0:i+1] + [javalang.tokenizer.JavaToken('\n', 
                    convert_char_offset_to_line_offset_java(idx, line_to_length)) 
                    for idx in new_line_index] + tokens[i+1:]
            token_range = token_range[0:i+1] + [(i, i+1, '\n') for i in new_line_index] + token_range[i+1:]
            i+=1
    
    if keep_string_only:
        tokens = [token.value for token in tokens]
    return tokens

def align_cubert_tokens_java(cubert_tokens, python_tokens, id=-1):
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
    #                     print(python_token, 'skipping', i,'tokens.', repr(python_token), remaining_cubert_tokens[i:i+length])	
    #                     print('remaining tokens', remaining_cubert_tokens[i+length:i+length+5])
                        result.append([*range(current_idx+i, current_idx+i+length)])
                        current_idx += length+i
                        found = True
                        break
            if not found:
                if python_token not in new_line:
                    print("id =", id, ": java token not found: ",  repr(python_token))
                result.append([])
            
        else:
            if python_token not in ignore_set and not python_token.isspace():
                print("ignoring java token: ", repr(python_token)) 
            result.append([])
        
    return result


tokenizer = java_tokenizer.JavaTokenizer()
subword_tokenizer = text_encoder.SubwordTextEncoder("cubert/vocab_java.txt")



# save subtoken alignment to json
with open('../../generating_CodeSyntax/deduplicated_java_code.pickle', 'rb') as f:
    sample = pickle.load(f)
examples = []
add_new_lines = False
output_file_name = "_with_new_lines" if add_new_lines else ""

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
                python_tokens = tokenize_java_code(sample[i], keep_string_only=True, add_new_lines=add_new_lines)
                if tokens == ['[CLS]_', '\\u\\u\\u', 'ERROR', '\\u\\u\\u_', '\\u\\u\\uNEWLINE\\u\\u\\u_', '[SEP]_']:
                    print("skipping", i, "because of cubert tokenization error")
                    continue
                alignment = align_cubert_tokens_java(tokens, python_tokens, id=i)
                examples.append({"tokens": tokens, "id": i, "alignment": alignment, "java_tokens": python_tokens})
        except Exception as e:
            print("skipping", i, "because of", e)
    print("length of batch", b, ":", len(examples))

    with open("data/CuBERT_tokenized/java_"+str(1000*b)+"_"+str(1000*(b+1))+output_file_name+".json", 'w') as f:
        json.dump(examples, f, indent=2)
        