import re

# script to convert stanford dependency to the format that preprocess_depparse.py requires
examples=[]
for i in range(1, 2455):
    try:
        with open("data/wsj_dependency/wsj_"+f'{i:04}'+".sd", "r") as f:
            lines = f.read().split('\n')
        examples.extend(lines)
        examples.append("")
    except Exception as e:
        print(e)


regex = re.compile(r'([a-z]+)\(.+-(\d+), (.+)-\d+\)')
formatted_examples = [] #  a word followed by a space followed by index_of_head-dependency_label (e.g., 0-root)

for line in examples:
    if line != "":
        mo = regex.search(line)
        label, index, word = mo.groups()
        formatted_examples.append(word+" "+index+"-"+label)
    else:
        formatted_examples.append("")


with open("data/deparse_english/dev.txt", "w") as f:
    f.write("\n".join(formatted_examples))