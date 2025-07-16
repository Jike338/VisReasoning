# data_utils.py
import json, os
from typing import Any
from datasets import Dataset, Image


def _json_to_str(val: Any) -> str:
    """Convert any non‑string to a JSON/str representation (prevents Arrow type errors)."""
    if isinstance(val, str):
        return val
    if isinstance(val, (int, float, bool)) or val is None:
        return str(val)
    return json.dumps(val, ensure_ascii=False, separators=(",", ":"))


def _folder_before_images(path: str) -> str:
    """
    Return the directory name immediately preceding 'images' in an absolute path.
    Example:
        /a/b/Thinklite/images/5977.png  ->  'Thinklite'
    """
    parts = os.path.normpath(path).split(os.sep)
    try:
        idx = parts.index("images")
        if idx > 0:
            return parts[idx - 1]
    except ValueError:
        pass
    # Fallback: parent folder of the file
    return os.path.basename(os.path.dirname(path))


def load_json_mixed_data(json_path: str) -> Dataset:
    """
    Load the custom mix‑data JSON (single JSON array) and return a Dataset with
    columns: decoded_image (lazy PIL.Image), question_prompt, answer, category.
    """

    # 1) read JSON array -> python list[dict]
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    # 2) clean / stringify fields & derive category
    cleaned = []
    for ex in raw:
        img_path = ex["image"]                    # keep path exactly as‑is
        ex_clean = {k: _json_to_str(v) for k, v in ex.items()}
        ex_clean["image"] = img_path
        ex_clean["category"] = _folder_before_images(img_path)
        cleaned.append(ex_clean)

    # 3) build Dataset
    ds = Dataset.from_list(cleaned)

    # 4) add lazy Image column & rename fields
    ds = ds.add_column("decoded_image", ds["image"]) \
           .cast_column("decoded_image", Image()) \
           .rename_column("problem", "question_prompt") \
           .rename_column("solution", "answer")

    return ds
