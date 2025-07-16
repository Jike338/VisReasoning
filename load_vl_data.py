from datasets import load_dataset
from PIL import Image as PILImage

MAX_PIXELS = 250_000
MIN_SIDE = 28

def _resize_image(img: PILImage.Image):
    w, h = img.size
    total = w * h

    if total > MAX_PIXELS:
        scale = (MAX_PIXELS / total) ** 0.5
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))
        img = img.resize((w, h), PILImage.LANCZOS)

    if w < MIN_SIDE or h < MIN_SIDE:
        w = max(w, MIN_SIDE)
        h = max(h, MIN_SIDE)
        img = img.resize((w, h), PILImage.LANCZOS)

    return img

def load_vis_reasoning(dataset_name_path, split="train"):
    dataset = load_dataset(dataset_name_path, split=split)

    def _process(example):
        example["question_prompt"] = example.get("question", "")
        example["solution"] = example.get("answer", "")
        img = example["image"]  # Already a PIL.Image.Image from datasets.Image()
        example["decoded_image"] = _resize_image(img)
        return example

    dataset = dataset.map(_process)
    dataset = dataset.remove_columns(["image"])
    
    return dataset
