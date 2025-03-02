#!/bin/bash
for fusion_type in "last" "static" "dynamic"; do
	echo "Running with fusion_type = $fusion_type"
	[ -d "$fusion_type.output" ] && rm -r "$fusion_type"
	python run_train.py --model_type smn --fusion_type "$fusion_type" \
		--do_train \
		--overwrite_output_dir \
		--per_gpu_train_batch_size 160 \
		--per_gpu_eval_batch_size 160 \
		--max_seq_length 50 \
		--learning_rate 0.001 \
		--num_train_epochs 1 \
		--data_dir ./data/pkl_files/ \
		--output_dir ./output/ \
		--num_embeddings 434511 \
		--embedding_size 200
	mv output "$fusion_type.output"
done
