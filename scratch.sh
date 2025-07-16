export VLLM_WORKER_MULTIPROC_METHOD=spawn
#!/bin/bash
#SBATCH --job-name=qwen-sft
#SBATCH --output=logs/qwen_sft_%j.out
#SBATCH --error=logs/qwen_sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=24:00:00
OpenGVLab/InternVL3-9B
llava-hf/llava-v1.6-mistral-7b-hf
llava-hf/llava-v1.6-vicuna-7b-hf

CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --dataset_name_path jike338/visreasoning2 \
    --model_name_path gpt-4o-mini \
    --duty_generation \
    --duty_extract_answer \
    --duty_calc_score \
    --bs 8 \
    --gen_engine vllm \
    --tag "" \
    --debug \
    --dataset_split train \
    --delete_prev_file \
    --gen_prompt_suffix_type "Please output the answer letter (A, B, C, D, E) or number (1, 2, 3) explicitly at the end." 

