import json
from collections import Counter

def extract_majority_vote_items(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    result = data.get("result", [])
    majority_vote_items = []

    for item in result:
        preds = item.get("normalized_pred", [])
        # Skip if preds is empty or contains nulls
        if not preds or any(p is None for p in preds):
            continue

        # Count frequencies and find majority
        count = Counter(preds)
        majority_answer, _ = count.most_common(1)[0]

        # Create a shallow copy of the item with updated normalized_pred
        item_copy = item.copy()
        item_copy["normalized_pred"] = [majority_answer]
        item_copy.pop("true_false")
        item_copy.pop("average_score")
        majority_vote_items.append(item_copy)

    return majority_vote_items

# Example usage
if __name__ == "__main__":
    input_path = "/scratch1/jikezhon/EVAL/results/debug/Qwen2.5-VL-7B-Instruct_score_vllm_dir_t1n10.json"  # Replace with the actual path
    results = extract_majority_vote_items(input_path)
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import score
    scorer = score.Scorer(results)
    scorer._calc_true_false()
    scorer._calc_per_sample_acc()