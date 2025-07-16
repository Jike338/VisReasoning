import json

def remove_last_4_keys_from_results(json_path, output_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data.get("result", []):
        keys = list(item.keys())
        for key in keys[-4:]:  # Remove last 4 keys
            item.pop(key, None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Example usage
remove_last_4_keys_from_results("/home/jikezhong/EVAL3.0/results/debug/visreasoning/InternVL3-9B_score_vllm_.json", "InternVL3-9B_raw_vllm_.json")
