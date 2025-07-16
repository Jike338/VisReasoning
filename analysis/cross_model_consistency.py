import json
from collections import defaultdict, Counter


# === Manually specify input files with ssh://usc to be stripped ===
raw_input_files = [
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_dir.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_cot.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_none.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_qwen.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_ans_then_reason.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_step_cot.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_ans_then_reason.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_cot.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_dir.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_none.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_qwen.json",
    "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_step_cot.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_hf_dir_t1n1.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_hf_dir_t1n1_0.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_hf_dir_t1n1_1.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_hf_dir_t1n1_2.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_hf_dir_t1n1_3.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_dir.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/InternVL3-8B-hf_score_hf_dir.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/InternVL3-9B_score_vllm_dir.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-32B-Instruct_score_hf_dir.json",
    # "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/llava-v1.6-vicuna-7b-hf_score_hf_dir.json"
]

# Strip "ssh://usc" from file paths
input_files = [path.replace("ssh://usc", "") for path in raw_input_files]
output_path = "merged_majority_vote.json"

# Step 1: Aggregate predictions and cache items
all_preds_by_pid = defaultdict(list)
items_by_pid = {}

# We'll copy these from the first file
score_section = None
parameters_section = None

for idx, fname in enumerate(input_files):
    with open(fname, "r") as f:
        data = json.load(f)
        if idx == 0:
            score_section = data.get("score", {})
            parameters_section = data.get("parameters", {})
        for item in data["result"]:
            pid = item["pid"]
            preds = item.get("normalized_pred", [])

            # Skip if the prediction is [null] or contains only None
            if not preds or all(p is None for p in preds):
                continue

            items_by_pid[pid] = item  # Latest non-null wins
            all_preds_by_pid[pid].extend([p for p in preds if p is not None])

# Step 2: Compute majority vote with deterministic tie-breaking
majority_voted_items = []
for pid, preds in all_preds_by_pid.items():
    if not preds:
        continue  # no valid prediction to vote on

    most_common = Counter(preds).most_common()
    max_count = most_common[0][1]
    top_candidates = [val for val, count in most_common if count == max_count]
    vote = sorted(top_candidates)[0]  # deterministic tie-breaking

    item = items_by_pid[pid]
    item["normalized_pred"] = [vote]
    item.pop("true_false", None)
    item.pop("average_score", None)
    majority_voted_items.append(item)

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import score
scorer = score.Scorer(majority_voted_items)
scorer._calc_true_false()
scorer._calc_per_sample_acc()

