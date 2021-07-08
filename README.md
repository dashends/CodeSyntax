# Benchmarking Language Models for Code Syntax Understanding

This is the repository for the EMNLP 2022 Findings paper "Benchmarking Language Models for Code Syntax Understanding." It contains:
1. The CodeSyntax dataset, a large-scale dataset of programs annotated with the syntactic relationships represented as edges in their corresponding  abstract  syntax  trees  (ASTs) (see folder CodeSyntax).
2. Code for building the CodeSyntax dataset (see folder generating_CodeSyntax).
3. Code for evaluating pre-trained language  models on code and natural language syntax understanding tasks (see folder evaluating_models).


## The CodeSyntax dataset
The folder CodeSyntax contains the following 5 dataset files in .zip compressed format.  
* The default python and java dataset, which we reported main results on:  
	* CodeSyntax_python.json  
	* CodeSyntax_java.json  
* The modified versions of python and java dataset that are used to conduct ablation study in section 4.2:  
	* CodeSyntax_python_with_new_lines.json and CodeSyntax_java_with_new_lines.json, where new line tokens are included in ground truth dependent nodes.  
	* CodeSyntax_java_skip_semicolon.json, where semicolon tokens are removed from ground truth dependent nodes.  

Each dataset file is a list of maps (one map for each code sample) in .json format. The maps have the following keys:  
* "code": original source code  
* "tokens": a list of code tokens (source code tokenized by python tokenize or javalang module)  
* "id": id of the sample   
* "relns": a map from relation names to a list of relation edges. Each relation edge is represented by a list of the form [head_index, dependent_start_index, dependent_end_index] where head_index is the index of the head token and the dependent block starts at dependent_start_index and ends at dependent_end_index (inclusively). Note that the indices start at 0.  
* For example, consider the following program that contains only one assignment statement and a relation edge from the 1-st token to the 3-rd token:  
&emsp;	 	 	 	 {"code": "q = queue"  
&emsp;	 	 	 	 "tokens": ["q", "=", "queue"]  
&emsp;	 	 	 	 "id": 1  
&emsp;	 	 	 	 "relns": {  
&emsp;&emsp;	 	 	 	 	 "Assign:target->value": [  
&emsp;&emsp;&emsp;	 	 	 	 	 	 [0,2,2]  
&emsp;&emsp;	 	 	 	 	 ]  
&emsp;	 	 	 	 }} 

##  Building the CodeSyntax dataset
The folder generating_CodeSyntax contains our code to generate the CodeSyntax dataset labeled with relations.
1. Install requirements:  
	* python 3.9  
	* python packages: pip install pandas, ast, javalang,  seaborn, matplotlib  
	* jupyter notebook  
	* Java SE 16 and Eclipse IDE  
2. Get and process source code from code search net:  
	*  	Run the dataset.ipynb notebook.  
3. Deduplicate (remove the code samples used in CuBERT and CodeBERT pre-training).  
	*  	Download CuBERT's pre-training dataset information [manifest.json](https://github.com/google-research/google-research/tree/master/cubert), place them in folder "Cubert Python" and "Cubert Java",  
	*	and then run the dataset.ipynb notebook.
4. Generate labels through AST parser:  
	* For python, we use the ast package as AST parser and the tokenize package as our tokenizer:  
		* 	python generate_labels_python.py  
	* For java, we use org.eclipse.jdt.core.dom's AST parser to get ast node's start and end positions and then feed it to python to convert position to token index using javalang tokenizer (which is the tokenizer used in CuBERT):  
		* 	Open Eclipse IDE.  
		* 	Click on import projects->general->import existing projects into workspace and choose the root folder generating_CodeSyntax\Java AST Parser.  
		* 	Build and run main.java.  
		* 	python generate_labels_java.py  
	* The generated dataset will be in the CodeSyntax folder.  
5. Generate dataset statistics  
	*  Run the last section of dataset.ipynb notebook.


## To extend CodeSyntax to another language
Follow the workflow discussed in the "building the CodeSyntax dataset" section. You need to find: a source code dataset, a tokenizer, and an AST parser for the target language, and substitube them into the existing framework.


## Evaluating language models
The folder evaluating_models contains our code for evaluating models and plotting results.

### To reproduce our results on programming languages  
Go to the folder evaluating_models/PL/ and then:  
(Note that running pre-trained language models and storing attention weights requires a significant amount of memory and disk space. If you would like to download our extracted attention weights needed to run the notebooks, the weights for the first 1000 samples are available here: https://drive.google.com/file/d/169yaIMSrCnzGQuBSc5wYMJ0F0ScqBScs/view?usp=sharing. You can download and unzip them into the folder PL/data/attention and then skip to step 8 or 9.)
1. Install requirements:
	*  To extract attention and evaluate models: 
		* Download and install [CuBERT](https://github.com/google-research/google-research/tree/master/cubert), [CodeBERT](https://github.com/microsoft/CodeBERT) and corresponding dependencies following their instructions.  Save pre-trained CuBERT models in the folder data/cubert_model_python and data/cubert_model_java.
		* Python packages: pip install transformers, tensorflow==1.15, torch, tensor2tensor, javalang,  numpy,  matplotlib  
	*  To plot results: 
		* Jupyter Notebook  
		* Python packages: pip install numpy, matplotlib  
2. Unzip dataset in the CodeSyntax folder.
3. Tokenize source code (code -> CuBERT/CodeBERT subtokens) and generate subtoken-token alignment:   
	*  python tokenize_and_align_cubert_java.py  
	*  python tokenize_and_align_cubert_python.py  
	*  python tokenize_and_align_codebert.py  
4. Run CuBERT/CodeBERT to extract attention and convert attention to word-level：
	*  bash run_exp_cubert_python.sh  
		If you save the model checkpoints at a different directory, you need to modify the --bert-dir argument in this .sh script.
	*  bash run_exp_cubert_python.sh  
	*  python run_exp_codebert_python.py  
	*  python run_exp_codebert_java.py
5. Preprocess attention (get predictions in the format of token index, sorted by weights)：  
	*  python preprocess_attn_python.py  
	*  python preprocess_attn_java.py  
	*  For CodeBERT, this step is already included in the previous step, i.e., run_exp_codebert_python.py.
6. Remove uncommon data points. CodeBERT tends to generate more subtokens. Sometimes CuBERT is able to process a sample (length <=512>), but CodeBERT can't (length>512).
	*  python remove_uncommon_datapoints.py
7. Generate top k scores for pre-trained models and baselines by running the following files:
	*  For CuBERT on Python dataset: topk_scores_cubert_attention.py
	*  For CuBERT on Java dataset: topk_scores_cubert_attention_java.py
	*  For CodeBERT on Python dataset: topk_scores_codebert_attention.py
	*  For CodeBERT on Java dataset: topk_scores_codebert_attention_java.py
	*  For baselines on Python dataset: topk_scores_baselines_python.py
	*  For baselines on Java dataset: topk_scores_baselines_java.py
8. Plot results and create tables:
	*  If you did not evaluate models by yourself, you need to unzip the file data/scores/scores.zip. Please place these .pkl data files in the folder data/scores.
	*  Run the notebook analysis.ipynb
9. Case study:
	*  Run the notebook case_study.ipynb

### To reproduce our results on natural languages  
Go to the folder evaluating_models/NL/ and then:  
(Note that we follow the paper "What Does BERT Look At? An Analysis of BERT's Attention" and utilize their code posted at https://github.com/clarkkev/attention-analysis. For more information about the attention-analysis subfolder, please refer to their repository.)
1. Download:
	*  [The English News Text Treebank: Penn Treebank Revised](https://catalog.ldc.upenn.edu/LDC2015T13) (not freely available). Unzip and place it in the folder data/eng_news_txt_tbnk-ptb_revised.
	*  [UD_German-HDT Hamburg Dependency Treebank](https://universaldependencies.org/#download). Unzip and place it in the folder data/ud-treebanks-v2.8_UD_German-HDT.
	*  [stanford parser](https://nlp.stanford.edu/software/lex-parser.shtml#Download)
	*  [BERT](https://github.com/google-research/bert)
	*  [RoBERTa](https://github.com/pytorch/fairseq/tree/main/examples/roberta)
	*  [CodeBERT](https://github.com/microsoft/CodeBERT)
	*  [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
	*  [XMR-RoBERTa](https://github.com/pytorch/fairseq/tree/main/examples/xlmr)
	*  pip install transformers, torch, tensorflow==1.15, matplotlib
3. Run stanford parser to convert the treebank into dependency labels. The results will be in the folder data/wsj_dependency.
	* bash treebank_to_dependency.sh
4. Convert dependency labels to the format that attention-analysis/preprocess_depparse.py requires.
	*  python convert_dependency_English.py  
	*  python convert_dependency_German.py  
	*  The results will be in the folder data/deparse_english and data/deparse_german
	*  sample results:  
	&emsp;&emsp;	Pierre 2-nn  
	&emsp;&emsp;	Vinken 9-nsubj  
	&emsp;&emsp;	, 2-punct  
	&emsp;&emsp;	61 5-num  
	&emsp;&emsp;	years 6-npadvmod  
	&emsp;&emsp;	old 2-amod  
	&emsp;&emsp;	, 2-punct  
	&emsp;&emsp;	...
4. Preprocess input data.  
	*  python ./attention-analysis/preprocess_depparse.py --data-dir data/depparse_english  
	*  python ./attention-analysis/preprocess_depparse.py --data-dir data/depparse_german  
	*  sample result of one sentence:
	&emsp;&emsp;	{"words": ["Pierre", "Vinken", ",", "61", "years", "old", ",", "will", "join", "the", "board", "as", "a", "nonexecutive", "director", "Nov.", "29", "."],   
	&emsp;&emsp;		"relns": ["nn", "nsubj", "punct", "num", "npadvmod", "amod", "punct", "aux", "root", "det", "dobj", "prep", "det", "amod", "pobj", "tmod", "num", "punct"],   
	&emsp;&emsp;		"heads": [2, 9, 2, 5, 6, 2, 2, 9, 0, 11, 9, 9, 15, 15, 12, 9, 16, 9]}
5. extract attention  
	*  python ./attention-analysis/extract_attention.py --preprocessed-data-file data/depparse_english/dev.json --bert-dir attention-analysis/cased_L-24_H-1024_A-16 --batch_size 4 --word_level --cased  
	*  python ./attention-analysis/extract_attention.py --preprocessed-data-file data/depparse_german/dev.json --bert-dir attention-analysis/multi_cased_L-12_H-768_A-12 --batch_size 4 --word_level --cased  
	*  If you placed pre-trained model checkpoint at a different location, please change the --bert-dir argument. For more information about extract_attention.py, please refer to https://github.com/clarkkev/attention-analysis.
	*  For RoBERTa and XLM-RoBERTa: python run_exp_roberta.py
6. Preprocess attention (get predictions in the format of token index, sorted by weights)： 
	*  preprocess_attn_NL_word_level_sorted.py for top k
7. Generate top k scores for attention and baseline:  
	*  topk_scores_attention_NL.py, topk_scores_baselines_NL.py
8. Plot results and create tables: 
	*  Run the notebook analysis.ipynb  
	   Note that the English treebank is not freely available, so we can not release the English dataset file dev.json.

## How to cite

```bibtex
@inproceedings{shen-etal-2022-codesyntax,
    title = "Benchmarking Language Models for Code Syntax Understanding",
    author = "Da Shen and Xinyun Chen and Chenguang Wang and Koushik Sen and Dawn Song",
    booktitle = "Findings of the Association for Computational Linguistics: {EMNLP} 2022",
    year = "2022",
    publisher = "Association for Computational Linguistics"
}
```


## Acknowledgements
* [What Does BERT Look At? An Analysis of BERT's Attention](https://github.com/clarkkev/attention-analysis)
* [Code Search Net](https://github.com/github/CodeSearchNet)
* [CuBERT](https://github.com/google-research/google-research/tree/master/cubert)
* [CodeBERT](https://github.com/microsoft/CodeBERT)