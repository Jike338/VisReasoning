import json
from collections import defaultdict

# Load the JSON data
with open("/home/jikezhong/EVAL3.0/results/debug/visreasoning/Qwen2.5-VL-7B-Instruct_score_vllm_.json", "r") as f:
    data = json.load(f)

# Target category
target_category = "3D Visualization / Cubes & Dice / Three views"

# Initialize counters
level_stats = defaultdict(lambda: {"correct": 0, "total": 0})

# Iterate through results
for item in data["result"]:
    if item["category"] == target_category:
        level = item["level"]
        is_correct = item["true_false"][0]
        level_stats[level]["total"] += 1
        if is_correct:
            level_stats[level]["correct"] += 1

# Calculate and display accuracy per level
print(f"Accuracy breakdown for category: '{target_category}'\n")
for level, stats in sorted(level_stats.items()):
    correct = stats["correct"]
    total = stats["total"]
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"Level {level}: {correct}/{total} correct ({acc:.2f}% accuracy)")
