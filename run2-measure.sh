#!/bin/bash
for fusion_type in "last" "static" "dynamic"; do
	echo "Running with fusion_type = $fusion_type"
	[ -d "$fusion_type.output" ] && rm -r "$fusion_type"
	python run_train.py --model_type smn --fusion_type "$fusion_type" \
		--do_train \
		--overwrite_output_dir \
		--per_gpu_train_batch_size 96 \
		--per_gpu_eval_batch_size 96 \
		--max_seq_length 50 \
		--learning_rate 0.001 \
		--num_train_epochs 1 \
		--data_dir ./data/pkl_files/ \
		--output_dir ./output/ \
		--num_embeddings 434511 \
		--embedding_size 200 &
	PY_PID=$!
	nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 1 > power_log-$fusion_type-$PY_PID.csv &
	POWER_PID=$!
	nvidia-smi pmon -s um -d 1 -o DT > gpu_utilization-$fusion_type-$PY_PID.log &
	PMON_PID=$!
	wait $PY_PID
	kill $POWER_PID $PMON_PID 2>/dev/null
	mv output "$fusion_type.output"
done
