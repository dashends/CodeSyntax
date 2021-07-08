from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaTokenizerFast
import re
import io
import pickle
import json
import tokenize,  javalang

def convert_line_offset_to_char_offset(line_offset, line_to_length):
    line, offset = line_offset
    if line > 1:
        offset += sum(line_to_length[:line-1]) + line-1
    return offset

# run tokenization and generate subtoken-token alignment
# Align cubert sub-tokens with python token
# Rewrite Attention-analysis’ tokenize_and_align(). 
# returns a list of lists where each sub-list 
# contains cubert-tokenized tokens for the correponding word.
# each cubert-subtoken is represented by int starting from 0
# e.g. [[1], [2, 3]]    where special token class is always at index 0

# By default, javalang tokenizer does not produce '\n' tokens while python tokenizer does.
# For Java, if we want to add '\n' into tokens, we need to produce different alignments too.

def align_codebert_tokens(codebert_encodings, python_tokens, code, id=-1):
    # get start and end of python tokens
    line_to_length = [len(l) for l in code.split("\n")] # stores the number of chars in each line of source code
    python_token_range = [(convert_line_offset_to_char_offset(token.start, line_to_length), 
                    convert_line_offset_to_char_offset(token.end, line_to_length), 
                    token.string) for token in python_tokens]

    # produce alignment by using start and end
    result = []
    for python_token in python_token_range:
        start, end, token = python_token
        codebert_token_index_first = codebert_encodings.char_to_token(start) # position in souce code -> token index
        codebert_token_index_last = codebert_encodings.char_to_token(end-1)
        if codebert_token_index_first != None and codebert_token_index_last!= None:
            result.append([*range(codebert_token_index_first, codebert_token_index_last+1)])
#             print(repr(token), codebert_encodings.tokens()[codebert_token_index_first: codebert_token_index_last+1])
        else:
            # not found
            if (not token.startswith('#') and not token.startswith('"""') and 
                    not token.startswith("'''") and token != "" and not token.isspace()):
                print("id =", id, ": python token not found: ",  repr(token))
            result.append([])
            
                
    return result


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

# run tokenization and generate subtoken-token alignment
# Align cubert sub-tokens with python token
# Rewrite Attention-analysis’ tokenize_and_align(). 
# returns a list of lists where each sub-list 
# contains cubert-tokenized tokens for the correponding word.
# each cubert-subtoken is represented by int starting from 0
# e.g. [[1], [2, 3]]    where special token class is always at index 0
def align_codebert_tokens_java(codebert_encodings, python_tokens, code, id=-1):
    # get start and end of python tokens
    line_to_length = [len(l) for l in code.split("\n")] # stores the number of chars in each line of source code
    python_token_range = [(convert_line_offset_to_char_offset_java(token.position, line_to_length), 
                    convert_line_offset_to_char_offset_java((token.position[0], token.position[1]+len(token.value)), line_to_length), 
                    token.value) for token in python_tokens]

    # produce alignment by using start and end
    result = []
    for python_token in python_token_range:
        start, end, token = python_token
        codebert_token_index_first = codebert_encodings.char_to_token(start)
        codebert_token_index_last = codebert_encodings.char_to_token(end-1)
        if codebert_token_index_first != None and codebert_token_index_last!= None:
            result.append([*range(codebert_token_index_first, codebert_token_index_last+1)])
#             print(repr(token), codebert_encodings.tokens()[codebert_token_index_first: codebert_token_index_last+1])
        else:
            # not found
            if (not token.isspace()):
                print("id =", id, ": java token not found: ",  repr(token))
            result.append([])
            
    return result



# cls code_tokens sep
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# python
with open('../../generating_CodeSyntax/deduplicated_python_code.pickle', 'rb') as f:
    sample = pickle.load(f)
examples = []
for i in range(len(sample)):
    if i%1000 == 0:
        print("processing sample ", i)
    encodings = tokenizer(sample[i])
    # class and sep special tokens are already added
    if len(encodings.tokens()) < 512:
        # generate alignment
        python_tokens = []
        for token in tokenize.generate_tokens(io.StringIO(sample[i]).readline):
            python_tokens.append(token)
        alignment = align_codebert_tokens(encodings, python_tokens, sample[i], i)
        python_tokens = [token.string for token in python_tokens]
        examples.append({'input_ids': encodings['input_ids'], "tokens": encodings.tokens(),
                             "id": i, "alignment": alignment, "python_tokens": python_tokens})
print(len(examples))

with open("data/CodeBERT_tokenized/python_full.json", 'w') as f:
    json.dump(examples, f, indent=2)


# java
# cls code_tokens sep
add_new_lines = False
output_file_name = "_with_new_lines" if add_new_lines else ""

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

with open('../../generating_CodeSyntax/deduplicated_java_code.pickle', 'rb') as f:
    sample = pickle.load(f)
    
has_unicode = []
for i in range(len(sample)):
    if '\\u' in sample[i]:
        has_unicode.append(i)
    
examples = []
for i in range(len(sample)):
    if i%1000 == 0:
        print("processing sample ", i)
    encodings = tokenizer(sample[i])
    # class and sep special tokens are already added
    if i not in has_unicode and len(encodings.tokens()) < 512:
        # generate alignment
        python_tokens = tokenize_java_code(sample[i], keep_string_only=False, add_new_lines=add_new_lines)
        alignment = align_codebert_tokens_java(encodings, python_tokens, sample[i], i)
        python_tokens = [token.value for token in python_tokens]
        examples.append({'input_ids': encodings['input_ids'], "tokens": encodings.tokens(),
                             "id": i, "alignment": alignment, "java_tokens": python_tokens})
print(len(examples))

with open("data/CodeBERT_tokenized/java_full"+output_file_name+".json", 'w') as f:
    json.dump(examples, f, indent=2)