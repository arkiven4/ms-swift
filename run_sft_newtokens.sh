PYTHONPATH=$PWD:$PYTHONPATH \
CUDA_HOME=/home/is/dwipraseetyo-a/nvidia/cuda-12.8/ \
HF_HOME=/home/is/dwipraseetyo-a/NAS_HAI/.cache \
MODELSCOPE_CACHE=/home/is/dwipraseetyo-a/NAS_HAI/.cache \
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
    --resume_from_checkpoint "outputs/250880-bigdata_balancediseasemodal-sft-llmalign-newtoken/v1-20250826-203637/checkpoint-1400" \
    --torch_dtype bfloat16 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
    --system prompts/grpo.txt \
    --dataset /home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets \
    --split_dataset_ratio 0.015 \
    --use_hf true \
    --trust_remote_code true \
    --datasets_script omnisft_dataset.py \
    --new_special_tokens 'plugins/new_tokens.txt' \
    --modules_to_save embed_tokens lm_head \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --attn_impl 'flash_attention_2' \
    --deepspeed 'zero2' \
    --eval_steps 250 \
    --save_steps 250 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --output_dir outputs/250880-bigdata_balancediseasemodal-sft-llmalign-newtoken \
    --overwrite_output_dir true
