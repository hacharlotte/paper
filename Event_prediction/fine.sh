# 17.2GiB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model deepseekv3/ \
    --model_type deepseek_r1_distill \
    --train_type lora \
    --resume_from_checkpoint 'output/deepseekv3/v23-20250320-152801/checkpoint-206' \
    --dataset 'data/processed_dataset_COT_R1.json' \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --model_author hacharlotte \
    --model_name sep-LLM