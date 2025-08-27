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
swift rlhf \
    --rlhf_type grpo \
    --model outputs/250880-bigdata_balancediseasemodal-sft-llmalign-newtoken/v5-20250827-023510/checkpoint-3000-merged \
    --reward_funcs external_r1v_acc format \
    --reward_weights 1 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset /home/is/dwipraseetyo-a/NAS_HAI/Datasets/grpo_3modalities_datasets \
    --use_hf true \
    --trust_remote_code true \
    --datasets_script "omnigrpo_dataset.py" \
    --system prompts/grpo.txt \
    --external_plugins /home/is/dwipraseetyo-a/NAS_HAI/Project/Qwen2.5-Omni/miscs/msswift_pluginsgrpo.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --bnb_4bit_compute_dtype bfloat16 \
    --bnb_4bit_use_double_quant true \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner true \
    --attn_impl 'flash_attention_2' \
    --deepspeed 'zero2' \
    --learning_rate 1e-5 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 16 \
    --dataset_num_proc 16 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.99 \
    --top_k 50 \
    --label_names solution \
    --output_dir outputs/250880-bigdata_balancediseasemodal-grpo-llm-newtoken \
    --overwrite_output_dir true \
    --log_completions true \
    --log_entropy true \
    --top_entropy_quantile 0.2 \
    --importance_sampling_level sequence \
    --epsilon 0.0003 \
    --epsilon_high 0.0004 \
    --beta 0
