python ./attention-analysis/extract_attention.py --preprocessed-data-file data/CuBERT_tokenized/python_0_1000.json --bert-dir data/cubert_model_python --max_sequence_length 512 --batch_size 4 --word_level
sleep 1

max=18
for (( i=1; i <= $max; ++i ))
do
	echo "${i}"
	python ./attention-analysis/extract_attention.py --preprocessed-data-file data/CuBERT_tokenized/python_${i}000_$((${i}+1))000.json --bert-dir data/cubert_model_python --max_sequence_length 512 --batch_size 4 --word_level
	sleep 1
done