PYTHONPATH=$PWD:$PYTHONPATH \
CUDA_HOME=/home/is/dwipraseetyo-a/nvidia/cuda-12.8/ \
HF_HOME=/home/is/dwipraseetyo-a/NAS_HAI/.cache \
RANK=0 \
WORLD_SIZE=1 \
GPUS_PER_NODE=1 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=3 \
OMP_NUM_THREADS=24 \
MAX_PIXELS=250880 \
ENABLE_AUDIO_OUTPUT=0 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --torch_dtype bfloat16 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --system /home/is/dwipraseetyo-a/NAS_HAI/Project/Qwen2.5-Omni/datas/sft_system.txt \
    --dataset /home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets \
    --split_dataset_ratio 0.01 \
    --use_hf true \
    --trust_remote_code true \
    --datasets_script "omnisft_dataset.py" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --attn_impl 'flash_attention_2' \
    --deepspeed 'zero2' \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --output_dir outputs/250880-bigdata-sft-llmalign \
    --overwrite_output_dir true
