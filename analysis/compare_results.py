import json
from pathlib import Path

def _clean_path(p: str) -> Path:
    """Remove leading 'ssh://usc' (if any) and return a Path object."""
    if "ssh://usc" in p:
        p = p.replace("ssh://usc", "", 1)  # drop only the first occurrence
    return Path(p)

def load_correctness(file_path: str) -> dict[str, bool]:
    """
    Return a mapping {pid: is_correct} for one result file.
    Assumes `true_false` is a list with a single boolean.
    """
    data = json.loads(_clean_path(file_path).read_text())
    return {
        item["pid"]: bool(
            item["true_false"][0] if isinstance(item["true_false"], list)
            else item["true_false"]
        )
        for item in data["result"]
    }

def compare_results(file_a: str, file_b: str) -> None:
    a = load_correctness(file_a)
    b = load_correctness(file_b)

    both_correct   = [pid for pid in a.keys() & b.keys() if a[pid] and b[pid]]
    only_a_correct = [pid for pid in a.keys() if a[pid] and (pid not in b or not b[pid])]
    only_b_correct = [pid for pid in b.keys() if b[pid] and (pid not in a or not a[pid])]
    both_wrong     = [pid for pid in a.keys() & b.keys() if not a[pid] and not b[pid]]

    # print("Correct in both:   ", both_correct)
    print(f"Correct only in A, total {len(only_a_correct)}: ", only_a_correct)
    print()
    print(f"Correct only in B, total {len(only_b_correct)}: ", only_b_correct)
    # print("Wrong in both:     ", both_wrong)

# ---- usage ----
f1 = "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_cot.json"
f2 = "ssh://usc/scratch1/jikezhon/EVAL/results/mathvista/Qwen2.5-VL-3B-Instruct_score_vllm_dir.json"
# f3 = "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2-VL-2B-Instruct_score_cot_hf.json"
# f4 = "ssh://usc/scratch1/jikezhon/EVAL/results/debug/Qwen2-VL-2B-Instruct_score_cot_vllm_bs1.json"

compare_results(f1, f2)
