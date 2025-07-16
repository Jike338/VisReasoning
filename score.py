from utils import normalize_extracted_answer, safe_equal
from tqdm import tqdm
from collections import defaultdict, OrderedDict

class Scorer:
    def __init__(self, items_with_extracted_responses, args=None, by_category=True):
        self.items_with_extracted_responses = items_with_extracted_responses
        self.by_category = by_category
        self.args = args
        self.scores = {}

    def _save(self):
        from utils import save_response_to_json
        save_response_to_json(self.args, self.items_with_extracted_responses, scores=self.scores)
    
    def _calc_per_sample_acc(self):
        correct = 0
        for item in tqdm(self.items_with_extracted_responses, desc="Scoring items..."):
            if 'average_score' in item:
                continue

            true_false_list = item['true_false']
            average_score = sum(true_false_list) / len(true_false_list)
            item['average_score'] = average_score
            correct+= average_score

        total = len(self.items_with_extracted_responses)
        accuracy = str(round(correct / total * 100, 2))
        self.scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}
        print(self.scores)

    def _calc_per_sample_acc_by_category(self):
        """
        Populates self.scores["by_category"] with
            {category_name: {"accuracy": str, "correct": float, "total": int}, ...}
        ordered by descending accuracy.
        """

        # ------------------------------------------------------------------
        # 1. Aggregate correct counts and totals per category
        # ------------------------------------------------------------------
        cat_stats = defaultdict(lambda: {"correct": 0.0, "total": 0})

        for item in self.items_with_extracted_responses:
            cat = item.get("category", "unknown")

            # average_score has already been filled by _calc_per_sample_acc()
            avg = item.get("average_score")
            if avg is None:                       # safety fallback
                true_false_all = item["true_false"]
                avg = sum(true_false_all) / len(true_false_all)

            cat_stats[cat]["correct"] += avg      # avg ∈ [0, 1]
            cat_stats[cat]["total"]   += 1

        # ------------------------------------------------------------------
        # 2. Convert to accuracy %
        # ------------------------------------------------------------------
        cat_scores = {}
        for cat, stats in cat_stats.items():
            acc = round(stats["correct"] / stats["total"] * 100, 2)
            cat_scores[str(cat)] = {
                "accuracy": str(acc),
                "correct":  stats["correct"],
                "total":    stats["total"],
            }

        # ------------------------------------------------------------------
        # 3. Sort by descending accuracy, then by category name
        # ------------------------------------------------------------------
        sorted_scores = OrderedDict(
            sorted(cat_scores.items(),
                   key=lambda kv: (-float(kv[1]["accuracy"]), kv[0]))
        )

        # ------------------------------------------------------------------
        # 4. Attach to the global score dict
        # ------------------------------------------------------------------
        self.scores["by_category"] = sorted_scores

        # Optional: quick console preview
        print("Accuracy by category (desc.):")
        for cat, stats in sorted_scores.items():
            print(f"  {cat:<15}  {stats['accuracy']}%  ({stats['correct']}/{stats['total']})")

    def _calc_true_false(self):
        for item in tqdm(self.items_with_extracted_responses, desc="calculating true false..."):
            if 'true_false' in item:
                continue
            answer = item['answer']
            pred_all = item['normalized_pred']
            true_false_all = []
            for pred in pred_all:
                true_false = safe_equal(pred, answer)
                true_false_all.append(true_false)

            item['true_false'] = true_false_all

    def _calculate_true_false_bbox_mix_data(self):
        for item in tqdm(self.items_with_extracted_responses, desc="calculating true false for bounding box..."):
            if 'true_false' in item:
                continue
            question = item['question_prompt']

            if "bounding box" in question.lower():
                gt = item['answer']
                pred_all = item['normalized_pred']
                true_false_all = []
                for pred in pred_all:
                    if pred == "none":
                        true_false_all.append(False)
                        continue
                    pred = [round(float(x)) for x in pred.strip("[]").split(",")]
                    gt = str(gt)
                    gt = [round(float(x)) for x in gt.strip("[]").split(",")]
                    true_false = self._iou(pred, gt)
                    true_false_all.append(true_false)
                item['true_false'] = true_false_all

    # needs to output true_false field
    def _get_normalized_preds_mathvista(self):
        for item in tqdm(self.items_with_extracted_responses, desc="getting normalized prediction..."):
            if 'normalized_pred' in item:
                continue

            choices = item['choices']
            question_type = item['question_type']
            answer_type = item['answer_type']
            extracted_responses = item['extracted_response']
            pred_all = []
            for extracted_response in extracted_responses:
                pred = normalize_extracted_answer(extracted_response, choices, question_type, answer_type)
                pred_all.append(pred)

            # update pred and true_false to the problem
            item['normalized_pred'] = pred_all

    def _get_normalized_preds_no_action(self):
        for item in tqdm(self.items_with_extracted_responses, desc="getting normalized prediction..."):
            if 'normalized_pred' in item:
                continue
            pred_all = []
            extracted_responses = item['extracted_response']
            for extracted_response in extracted_responses:
                pred_all.append(extracted_response)
            # update pred and true_false to the problem
            item['normalized_pred'] = pred_all
            
    def _get_normalized_preds_int(self):
        for item in tqdm(self.items_with_extracted_responses, desc="getting normalized prediction..."):
            if 'normalized_pred' in item:
                continue

            extracted_responses = item['extracted_response']
            pred_all = []
            for resp in extracted_responses:
                try:
                    pred = int(float(resp))  # Handles both "6" and "6.0"
                except (ValueError, TypeError):
                    pred = -1
                pred_all.append(pred)

            item['normalized_pred'] = pred_all

    def _iou(self, box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2] - 1, box2[2] - 1)
        inter_y2 = min(box1[3] - 1, box2[3] - 1)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
        else:
            inter = 0
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        return float(inter) / union

    def _extend_pred_w_underscore(self):
        for item in tqdm(self.items_with_extracted_responses, desc="getting normalized prediction..."):
            normalized_preds = item['normalized_pred']
            normalized_preds_underscore = []
            for prediction in normalized_preds:
                prediction = prediction.strip().lower().replace(" ", "_") if prediction else None
                normalized_preds_underscore.append(prediction)
            # update pred and true_false to the problem
            item['normalized_pred'] = normalized_preds_underscore

    def calc_score_and_save(self):

        if "mathvista" in self.args.task_name.lower():
            self._get_normalized_preds_mathvista()
        elif "superclevr_counting" in self.args.task_name.lower():
            self._get_normalized_preds_int()
        elif "mix_data" in self.args.task_name.lower():
            self._get_normalized_preds_no_action()
            self._calculate_true_false_bbox_mix_data()
        elif "hi_tom" in self.args.task_name.lower():
            self._get_normalized_preds_no_action()
            self._extend_pred_w_underscore()
        else:
            self._get_normalized_preds_no_action()

        self._calc_true_false()
        self._calc_per_sample_acc()
        self._calc_per_sample_acc_by_category()
        self.args.duty_type = "score"
        self._save()