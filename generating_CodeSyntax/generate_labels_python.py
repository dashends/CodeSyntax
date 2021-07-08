import pickle
import ast
import tokenize
import io
import json
from collections import defaultdict





class Visit(ast.NodeVisitor):
    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self.result = defaultdict(lambda : []) # relation -> list of tuples (dep_index, dependent_start_index, dependent_end_index) 
        # self.op_dict = {ast.And: "and", ast.Or: "or", ast.Not: "not", ast.Invert:"~", ast.UAdd: "+", ast.USub: "-"}
        # self.all_relns = set()
            
    def set_code_and_tokens(self, i):
        self.code = examples[i]["code"]
        self.tokens = examples[i]["tokens"]
        
    def clear_result(self):
        self.result = defaultdict(lambda : [])
        
    def generic_visit(self, node):
        # try:
            name = type(node).__name__
            if name == 'If':
                # "if-else":
                if len(node.orelse) != 0:
                    # else or elif token will be within this range:
                    start_idx = self.get_idx(node.body[-1], first_token=False)
                    end_idx = self.get_idx(node.orelse[0])
                    for i in range(start_idx+1, end_idx+1):
                        if self.tokens[i] == "else" or self.tokens[i] == "elif":
                            self.result["If:if->else"].append((self.get_idx(node), i, i))
                            break

                self.add_idx_tuple("If:if->test", node, node.test, node.test)
                self.add_idx_tuple("If:if->body", node, node.body[0], node.body[-1])

                # If(test, body, orelse)
                # if-test-body
                self.add_idx_tuple("If:test->body", node.test, node.body[0], node.body[-1])
                # if-test-orelse
                if len(node.orelse) != 0:
                    self.add_idx_tuple("If:test->orelse", node.test, node.orelse[0], node.orelse[-1])
                # if-body-orelse
                if len(node.orelse) != 0:
                    self.add_idx_tuple("If:body->orelse", node.body[0], node.orelse[0], node.orelse[-1])
            elif name == 'Call':
                # Call(func, args, keywords, starargs, kwargs)
                # But starargs and kwargs do not exist in nodes
                arg = None
                if len(node.args) != 0 and node.args[0]!= None:
                    self.add_idx_tuple("Call:func->args", node.func, node.args[0], node.args[-1], head_first_token=False)
                    if len(node.keywords) != 0 and node.keywords[0]!= None:
                        self.add_idx_tuple("Call:args->keywords", node.args[0], node.keywords[0], node.keywords[-1], head_first_token=False) 
                if len(node.keywords) != 0 and node.keywords[0]!= None:
                    self.add_idx_tuple("Call:func->keywords", node.func, node.keywords[0], node.keywords[-1], head_first_token=False)
            elif name == 'Assign':
                # if self.reln == "assignment-target-value":
                    self.add_idx_tuple("Assign:target->value", node.targets[0], node.value, node.value)
            elif name == 'While':
                self.add_idx_tuple("While:while->test", node, node.test, node.test)
                self.add_idx_tuple("While:while->body", node, node.body[0], node.body[-1])
                # While(test, body, orelse)
                self.add_idx_tuple("While:test->body", node.test, node.body[0], node.body[-1])

            elif name == 'For': 
                self.add_idx_tuple("For:for->body", node, node.body[0], node.body[-1])
                self.add_idx_tuple("For:for->target", node, node.target, node.target)
                self.add_idx_tuple("For:for->iter", node, node.iter, node.iter)

                #  For(target, iter, body, orelse)
                self.add_idx_tuple("For:target->iter", node.target, node.iter, node.iter)
                self.add_idx_tuple("For:target->body", node.target, node.body[0], node.body[-1])
                self.add_idx_tuple("For:iter->body", node.iter, node.body[0], node.body[-1])

            elif name == 'BinOp': 
                # BinOP(left, op, right)
                self.add_idx_tuple("BinOp:left->right", node.left, node.right, node.right)
            elif name == 'BoolOp':
                # BoolOp(op, values)
                # self.add_idx_tuple("BoolOp:op->value", node.op, node.values[0])
                if len(node.values) >= 2:
                    self.add_idx_tuple("BoolOp:value->value", node.values[0], node.values[1], node.values[1])
            elif name == 'Compare': 
                # Compare(left, ops, comparators)
                self.add_idx_tuple("Compare:left->comparator", node.left, node.comparators[0], node.comparators[-1])
                # self.add_idx_tuple("Compare:left->op", node.left, node.ops[0])
                # self.add_idx_tuple("Compare:op->comparator", node.ops[0], node.comparators[0])

            elif name == 'Try': 
                #  Try(body, handlers, orelse, finalbody)
                if len(node.handlers) != 0 and node.handlers[0] != None:
                    self.add_idx_tuple("Try:body->handler", node.body[0], node.handlers[0], node.handlers[-1])

                    if len(node.orelse) != 0 and node.orelse[0] != None:
                        self.add_idx_tuple("Try:handler->orelse", node.handlers[0], node.orelse[0], node.orelse[0])
                    if len(node.finalbody) != 0 and node.finalbody[0] != None:
                        self.add_idx_tuple("Try:handler->finalbody", node.handlers[0], node.finalbody[0], node.finalbody[-1])
                if len(node.orelse) != 0 and node.orelse[0] != None:
                    self.add_idx_tuple("Try:body->orelse", node.body[0], node.orelse[0], node.orelse[-1])
                if len(node.finalbody) != 0 and node.finalbody[0] != None:
                    self.add_idx_tuple("Try:body->finalbody", node.body[0], node.finalbody[0], node.finalbody[-1])
            
            elif name == 'IfExp':
                # IfExp(test, body, orelse)
                self.add_idx_tuple("IfExp:body->test", node.body, node.test, node.test)
                if node.orelse != None:
                    self.add_idx_tuple("IfExp:test->orelse", node.test, node.orelse, node.orelse)
                if node.orelse != None:
                    self.add_idx_tuple("IfExp:body->orelse", node.body, node.orelse, node.orelse)

            elif name == 'Attribute':
                # Attribute(value, attr, ctx)
                self.add_idx_tuple("Attribute:value->attr", node.value, node, node, dep_first_token=False)

            elif name == 'Dict':
                # Dict(keys, values)
                for i in range(len(node.keys)):
                    if node.keys[i] != None and node.values[i] != None:
                        self.add_idx_tuple("Dict:key->value", node.keys[i], node.values[i], node.values[i])

            elif name == 'Subscript':
                # Subscript(value, slice, ctx)
                self.add_idx_tuple("Subscript:value->slice", node.value, node.slice, node.slice)
            
            elif name == 'Slice':
                # Slice(lower, upper, step)
                self.add_idx_tuple("Slice:lower->upper", node.lower, node.upper, node.upper)
            
            elif name == 'AugAssign':
                # AugAssign(target, op, value)
                self.add_idx_tuple("AugAssign:target->value", node.target, node.value, node.value)
            
            elif name == 'With':
                # With(items, body, type_comment)
                self.add_idx_tuple("With:item->body", node.items[0].context_expr, node.body[0], node.body[-1])
            
            elif name == 'ListComp' or  name == 'SetComp' or  name == 'GeneratorExp':
                # ListComp(elt, generators), SetComp(elt, generators), GeneratorExp(elt, generators)
                self.add_idx_tuple(name+":elt->generator", node.elt, node.generators[0].target, node.generators[-1].iter)

            elif name == 'DictComp':
                # DictComp(key, value, generators) 
                self.add_idx_tuple(name+":key->value", node.key, node.value, node.value)
                self.add_idx_tuple(name+":key->generator", node.key, node.generators[0].target, node.generators[-1].iter)
                self.add_idx_tuple(name+":value->generator", node.value, node.generators[0].target, node.generators[-1].iter)
            
            elif name == 'comprehension':
                # comprehension(target, iter, ifs, is_async)
                self.add_idx_tuple(name+":target->iter", node.target, node.iter, node.iter)
                
            elif name == 'FunctionDef' or name == 'AsyncFunctionDef':
                for item in node.body:
                    self.visit(item)
                return

    #         ast.NodeVisitor.generic_visit(self, node)
            # super class's generic visit function, but add else if orelse is not empty
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.add_child_tuple(node, item)
                            self.visit(item)
                elif isinstance(value, ast.AST):
                    self.add_child_tuple(node, value)
                    self.visit(value)
        # except AttributeError as e:
        #     print(e)
        #     print(sample[i])
        #     print(node)
        #     exit()  


    def add_child_tuple(self, parent, child):
        if (parent != None and child != None and hasattr(parent, 'lineno') and hasattr(child, 'lineno') 
                and type(parent).__name__ != 'FunctionDef' and 
                type(parent).__name__ != 'AsyncFunctionDef' and type(child).__name__ != 'FunctionDef' and 
                type(child).__name__ != 'AsyncFunctionDef'):
            parent_idx = self.get_idx(parent)
            child_idx = self.get_idx(child)
            child_end_idx = self.get_idx(child, False)
            if parent_idx != child_idx and parent_idx > 0 and child_idx > 0 and parent_idx < child_idx and  child_idx <= child_end_idx:
                self.result["children:parent->child"].append((parent_idx, child_idx, child_end_idx))

    def add_idx_tuple(self, reln, head_ast, dep_ast, dep_end_ast, head_first_token=True, dep_first_token=True):
        # create the tuple (head_idx, dep_idx) from ast nodes head->dep
        if head_ast != None and dep_ast!= None:
            head_idx = self.get_idx(head_ast, head_first_token)
            dep_idx = self.get_idx(dep_ast, dep_first_token)
            dep_end_idx = self.get_idx(dep_end_ast, False)
            if head_idx != dep_idx and head_idx > 0 and dep_idx > 0 and head_idx < dep_idx and  dep_idx<= dep_end_idx: 
                # if self.tokens[dep_end_idx] == '\n':
                #     print(reln, self.tokens[head_idx:dep_end_idx+1])
                self.result[reln].append((head_idx, dep_idx, dep_end_idx))
        return

    def get_matching_tokens_count(self, tokens, code, indent_added=False):
        # return the number of tokens that starts in the beginning and are the same

        tokens2 = []
        try:
            for token in tokenize.generate_tokens(io.StringIO(code).readline):
                tokens2.append(token.string)
        except Exception as e:
            # print(e, code)
            ...

        if indent_added and tokens2[0].isspace():
            tokens2 = tokens2[1:]

        for i in range(min(len(tokens), len(tokens2))):
            if tokens[i] != tokens2[i]:
                # the previous i tokens are matched
                return i
        return min(len(tokens), len(tokens2))

    def get_idx(self, node, first_token=True):
        # get the index of the first/last token of ast_node in python_tokens
        # if first_token == true, get the index for the first token
        
        if isinstance(node, list):
            node = node[0] if first_token else node[-1]
        
        # ast.comprehension object has no attribute "lineno"
        if type(node).__name__ == "comprehension":
            node = node.target if first_token else node.iter

        
        # get token count in previous segments
        code_prev = ("\n").join(self.code.split("\n")[0:node.lineno-1]) +"\n"+ self.code.split("\n")[node.lineno-1][0:node.col_offset] + "_EndOfLineSymbol_"
        
        
        # only account for macthing tokens and ignore extra ending strings
        count_prev = self.get_matching_tokens_count(self.tokens, code_prev)

        # get token count in current segment
        if first_token:
            count_curr = 1
            # exclude '(' and ')'  from results
            while self.tokens[count_prev+count_curr-1] in left_parens and self.tokens[count_prev+count_curr] not in right_parens:
                count_curr += 1
        else:
            code_curr = ast.get_source_segment(self.code , node)
            # to prevent indentation error, we need to add previous blank space 
            previous_token = self.code.split("\n")[node.lineno-1][0:node.col_offset]
            if previous_token.isspace():
                code_curr = previous_token + code_curr
                count_curr = self.get_matching_tokens_count(self.tokens[count_prev:], code_curr, indent_added=True)
            else:
                count_curr = self.get_matching_tokens_count(self.tokens[count_prev:], code_curr)
            # add new lines
            if add_new_lines and count_prev+count_curr < len(self.tokens) and self.tokens[count_prev+count_curr] == '\n':
                count_curr += 1
            # exclude '(' and ')'  from results
            while self.tokens[count_prev+count_curr-1] in right_parens:
                count_curr -= 1

        
        return count_prev+count_curr-1
   

def get_label(i, visitor):
    ast_node = ast.parse(examples[i]["code"])
    visitor.set_code_and_tokens(i)
    visitor.visit(ast_node)
    result = dict(visitor.result)
    visitor.clear_result()
    return result



add_new_lines = False
left_parens = set(['(', '[', '{'])
right_parens = set([')', ']', '}'])


with open('deduplicated_python_code.pickle', 'rb') as f:
    sample = pickle.load(f)

examples = []
print("tokenizing python code")
for i in range(0, len(sample)):
    python_tokens = []
    for token in tokenize.generate_tokens(io.StringIO(sample[i]).readline):
        python_tokens.append(token.string)
    examples.append({"tokens": python_tokens, "id": i, "code": sample[i], 'relns': {}})



print("generating labels")
error_count = 0
total_count = 0
v = Visit()
for i in range(len(examples)):
# for i in [6]:
    if i%100 == 0:
        print("processing", i)
    total_count += 1
    try:
        examples[i]['relns'] = get_label(i, v) # list of tuple [(head_idx, dep_idx), ...]
    except SyntaxError as e:
        print("skipping example ", i, "due to syntax error ")
        error_count += 1
    except Exception as e:
        error_count += 1
        print("skipping example ", i, "due to error ", e)



print("error_count", error_count)
print("total_count", total_count)

output_file_name = "CodeSyntax_python"
if add_new_lines:
    output_file_name += "_with_new_lines"
with open("../CodeSyntax/"+output_file_name+".json", 'w') as f:
    json.dump(examples, f, indent=2)

# print(examples[19])