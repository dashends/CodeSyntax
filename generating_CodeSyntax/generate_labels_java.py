import pickle
import json
from collections import defaultdict
import javalang
import re

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
    
class Visit():
    def __init__(self):
        self.result = defaultdict(lambda : []) # relation -> list of tuples [(head_idx, dep_idx), ...]
        # self.op_dict = {ast.And: "and", ast.Or: "or", ast.Not: "not", ast.Invert:"~", ast.UAdd: "+", ast.USub: "-"}
        # self.all_relns = set()
            
    def set_code_and_tokens(self, i):
        self.code = examples[i]["code"]
        self.tokens = examples[i]["tokens"]
        
    def clear_result(self):
        self.result = defaultdict(lambda : [])

    def parse_positions(self, label_and_positions):
        for label_and_position in label_and_positions:
            reln, head_start, head_end, dep_start, dep_end = label_and_position.split(" ")
            self.add_idx_tuple(reln, int(head_start), int(head_end), int(dep_start), int(dep_end))

    def add_idx_tuple(self, reln, head_start, head_end, dep_start, dep_end, head_first_token=True, dep_first_token=True):
        # create the tuple (head_idx, dep_idx)
        head_idx = self.get_idx(head_start, head_end, True)
        dep_start_idx = self.get_idx(dep_start, dep_end, True)
        dep_end_idx = self.get_idx(dep_start, dep_end, False)
        if (head_idx != None and dep_start_idx!= None and dep_end_idx != None and 
                head_idx != dep_start_idx and head_idx > 0 and dep_start_idx > 0 
                and head_idx < dep_start_idx and  dep_start_idx<= dep_end_idx): 
            self.result[reln].append((head_idx, dep_start_idx, dep_end_idx))
      

    def get_matching_tokens_count(self, tokens, code):
        # return the number of tokens that starts in the beginning and are the same
        tokens2 = tokenize_java_code(code, keep_string_only=True, add_new_lines=add_new_lines)

        for i in range(min(len(tokens), len(tokens2))):
            if tokens[i] != tokens2[i]:
                # the previous i tokens are matched
                return i
        return min(len(tokens), len(tokens2))

    def get_idx(self, start_position, end_position, first_token=True):
        # get the index of the first/last token within the range between start_position and end_position
        # if first_token == true, get the index for the first token
        
        # get token count in previous segments
        code_prev = self.code[0:start_position] + "_EndOfLineSymbol_"
        
        
        # only account for macthing tokens and ignore extra ending strings
        count_prev = self.get_matching_tokens_count(self.tokens, code_prev)

        # get token count in current segment
        if first_token:
            count_curr = 1
            # exclude '(' and ')'  from results
            while self.tokens[count_prev+count_curr-1] in skip_set[0]:
                count_curr += 1
                # if we reach the end of the block, don't add this label
                if self.tokens[count_prev+count_curr-1] in skip_set[1]:
                    return None
        else:
            code_curr = self.code[start_position:end_position]
            count_curr = self.get_matching_tokens_count(self.tokens[count_prev:], code_curr)
            # add new lines
            if add_new_lines and count_prev+count_curr < len(self.tokens) and self.tokens[count_prev+count_curr] == '\n':
                count_curr += 1
            # exclude '(' and ')'  from results
            while self.tokens[count_prev+count_curr-1] in skip_set[1]:
                count_curr -= 1
                if self.tokens[count_prev+count_curr-1] in skip_set[0]:
                    return None
        return count_prev+count_curr-1
   

def get_label(i, visitor, positions):
    visitor.set_code_and_tokens(i)
    visitor.parse_positions(positions)
    result = dict(visitor.result)
    visitor.clear_result()
    return result






Skip_semicolon = False
add_new_lines = True
assert not (add_new_lines == True and Skip_semicolon == True)



with open('deduplicated_java_code.pickle', 'rb') as f:
    sample = pickle.load(f)

examples = []
print("tokenizing java code")
for i in range(0, len(sample)):
    java_tokens = tokenize_java_code(sample[i], keep_string_only=True, add_new_lines=add_new_lines)
    examples.append({"tokens": java_tokens, "id": i, "code": sample[i], 'relns': {}})



print("generating labels")
error_count = 0
total_count = 0
v = Visit()

# get start and end positions for each label and node
file = open('Java AST Parser/java_node_start_end_position.txt',mode='r')
positions = file.read().split("\n")
file.close()
id_to_positions = {}
current_id = 0
cache = []
for line in positions:
    if line.isdigit():
        id_to_positions[current_id] = cache
        cache = []
        current_id = int(line)
    elif line != "":
        cache.append(line)
id_to_positions[current_id] = cache


# generate labels
skip_set = (set(["(","[","{","\n"]), set([")","]","}", ";"])) if Skip_semicolon else  (set(["(","[","{","\n"]), set([")","]","}"]))
time_prev = None
for i in range(len(examples)):
    if i%100 == 0:
        print("processing", i)
    total_count += 1    
    examples[i]['relns'] = get_label(i, v, id_to_positions[i]) # list of tuple [(head_idx, dep_idx), ...]




print("error_count", error_count)
print("total_count", total_count)

output_file_name = "CodeSyntax_java"
if add_new_lines:
    output_file_name += "_with_new_lines"
elif Skip_semicolon:
    output_file_name += "_skip_semicolon"
with open("../CodeSyntax/"+output_file_name+".json", 'w') as f:
    json.dump(examples, f, indent=2)