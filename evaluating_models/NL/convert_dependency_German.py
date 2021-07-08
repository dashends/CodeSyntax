# script to convert universal dependency to the format that preprocess_depparse.py requires
examples=[]
for partition in ["test", "dev", "train"]:
    with open("data/ud-treebanks-v2.8_UD_German-HDT/de_hdt-ud-"+partition+".conllu", "r", encoding='utf-8') as f:
        lines = f.read().split('\n')
    examples.extend(lines)


formatted_examples = [] #  a word followed by a space followed by index_of_head-dependency_label (e.g., 0-root)
count = 0

for line in examples:
    if line != "" and not line.startswith("#"):
        words = line.split("\t")
        word, index, label  = words[1], words[6], words[7]
        formatted_examples.append(word+" "+index+"-"+label)
    elif line == "":
        count += 1
        formatted_examples.append("")

print("number of examples:", count)


with open("data/depparse_german/dev.txt", "w", encoding='utf-8') as f:
    f.write("\n".join(formatted_examples))