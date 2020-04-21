CUDA_VISIBLE_DEVICES=1 python modules/run_classifier.py --data_dir ../data/psyc_MISC11_ML_17_padding \
--model_type bert \
--config_name bert-base-cased \
--tokenizer_name DeepPavlov/bert-base-cased-conversational \
--model_name_or_path DeepPavlov/bert-base-cased-conversational \
--task_name categorize-both-speaker-conversational-bert \
--num_train_epochs 20 \
--output_dir ../results/categorize-both-speaker-conversational-bert \
--local_rank -1 \
--cache_dir ../../../.cache \
--per_gpu_train_batch_size 8 \
--per_gpu_eval_batch_size 16 \
--gradient_accumulation_steps 8 \
--save_steps 7500 \
--warmup_steps 2000 \
--weight_decay 0.1 \
--learning_rate 3e-5 \
--do_train \
--overwrite_output_dir

# --do_train \
# --evaluate_during_training \
# --overwrite_output_dir

# --do_eval \
# --eval_all_checkpoints

# --do_test \
# --eval_all_checkpoints
