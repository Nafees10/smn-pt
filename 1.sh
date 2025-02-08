#SMN-last
python run_train.py --model_type smn --fusion_type last --do_train --overwrite_output_dir --per_gpu_train_batch_size 200 --per_gpu_eval_batch_size 200 --max_seq_length 50 --learning_rate 0.001 --num_train_epochs 10 --data_dir ./data/pkl_files/ --output_dir ./output/ --num_embeddings 434511
