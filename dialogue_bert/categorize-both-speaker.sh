python -m pdb -c continue modules/run_classifier.py --data_dir ../data/psyc_MISC11_ML_17_padding \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name categorize-both-speaker-dialogue-bert \
--num_train_epochs 10 \
--output_dir ../results/categorize-both-speaker-dialogue-bert \
--max_seq_length=512 \
--local_rank -1 \
--cache_dir ../../../.cache \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--save_steps 8000 \
--warmup_steps 2000 \
--weight_decay 0.1 \
--learning_rate 3e-5 \
--do_test

# --do_train \
# --evaluate_during_training \
# --overwrite_output_dir

# --do_eval \
# --eval_all_checkpoints

# --do_test \
# --eval_all_checkpoints
