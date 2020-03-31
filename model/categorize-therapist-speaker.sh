python -m pdb -c continue modules/run_classifier.py --data_dir ../data/psyc_MISC11_ML_17_padding \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name categorize-therapist-speaker \
--num_train_epochs 7 \
--output_dir ../results/categorize-therapist-speaker \
--max_seq_length=512 \
--local_rank -1 \
--cache_dir ../../../.cache \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 32 \
--gradient_accumulation_steps 2 \
--warmup_steps 400 \
--do_test \
--eval_all_checkpoints

# --do_train \
# --evaluate_during_training \
# --overwrite_output_dir

# --do_eval \
# --eval_all_checkpoints

# --do_test \
# --eval_all_checkpoints
