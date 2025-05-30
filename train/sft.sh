#!/bin/bash

# Reference Running: bash train/sft.sh
# {'train_runtime': 5268.8407, 'train_samples_per_second': 0.949, 'train_steps_per_second': 0.119, 'train_loss': 0.1172730620391667, 'epoch': 5.0}
uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-0.5B-Instruct"
router_cutoff_layer=6
num_loras=4
lora_r=128
lora_alpha=256

# lr=1e-5
lr=3e-4
min_lr=1e-6
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
# gpu_count=$(nvidia-smi -L | wc -l)
gpu_count=1
push_to_hub=false

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_lora.py \
    --block_size=8192 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --optim="adamw_bnb_8bit" \
    --output_dir="ckpts/s1-${uid}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --gradient_checkpointing=False \
    --wandb_project="s1" \
    --wandb_entity="quang-vn-university-of-engineering-and-technology-vnu"
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
    # --train_file_path="simplescaling/splited_tokenized/split_1_of_5" \
    # --fsdp="full_shard auto_wrap" \
    # --fsdp_config="train/fsdp_config_qwen.json" \
