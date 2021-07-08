for file in ../data/eng_news_txt_tbnk-ptb_revised/data/penntree/**/*.tree; do
	echo ${file:55:8}
	java -mx1g edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -treeFile ${file} > ../data/wsj_dependency/${file:55:8}.sd
done