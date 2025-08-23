PYTHONPATH=$PWD:$PYTHONPATH \
RANK=0 \
WORLD_SIZE=1 \
CUDA_HOME=/opt/nvidia/cuda-12.2u2 \
HF_HOME=/work/dwipraseetyo-a/.cache \
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=24 \
MAX_PIXELS=250880 \
ENABLE_AUDIO_OUTPUT=0 \
GPUS_PER_NODE=2 \
NPROC_PER_NODE=2 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-Omni-7B \
    --reward_funcs external_r1v_acc format \
    --reward_weights 1 0.5 \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --dataset /work/dwipraseetyo-a/Datasets/grpo_3modalities_datasets \
    --use_hf true \
    --trust_remote_code true \
    --system /work/dwipraseetyo-a/Qwen2.5-Omni/datas/grpo_system.txt \
    --external_plugins /work/dwipraseetyo-a/Qwen2.5-Omni/miscs/msswift_pluginsgrpo.py \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    # --quant_method bnb \
    # --quant_bits 4 \
    # --bnb_4bit_quant_type nf4 \
    # --bnb_4bit_compute_dtype bfloat16 \
    # --bnb_4bit_use_double_quant true \
    --freeze_vit true \
    --freeze_llm false \
    --freeze_aligner false \
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
    --num_generations 2 \
    --temperature 1.0 \
    --top_p 0.99 \
    --top_k 50 \
    --label_names solution \
    --output_dir outputs/250880-bigdata-grpo-llm \
    --overwrite_output_dir true \
    --log_completions true \
    --log_entropy true
    # --top_entropy_quantile 0.2 \
    # --importance_sampling_level sequence \
    # --epsilon 0.0003 \
    # --epsilon_high 0.0004 \
    # --beta 0 \
