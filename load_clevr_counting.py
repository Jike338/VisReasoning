# data_utils.py
import os
from typing import Union

from datasets import load_dataset, Dataset, Image


def load_jsonl_with_images_superclever_counting(
    jsonl_path: str,
    data_dir: Union[str, os.PathLike],
    split_name: str = "test",
) -> Dataset:

    # ---------------------------------------------------------------------
    # 1) Raw load (no image decoding yet)
    # ---------------------------------------------------------------------
    ds = load_dataset(
        "json",
        data_files={split_name: jsonl_path},  # explicit split name
        split=split_name,
        streaming=False,
    )

    # ---------------------------------------------------------------------
    # 2) Rewrite relative paths -> absolute/desired paths
    # ---------------------------------------------------------------------
    def _rewrite_path(example):
        file_name = os.path.basename(example["image_path"])
        example["image_path"] = os.path.join(os.fspath(data_dir), file_name)
        return example

    ds = ds.map(_rewrite_path)

    # ---------------------------------------------------------------------
    # 3) Cast *existing* `image_path` column to the Arrow Image() feature
    #    This keeps decoding lazy and avoids creating a dummy column.
    # ---------------------------------------------------------------------
    ds = ds.add_column("image", ds["image_path"])
    ds = ds.cast_column("image", Image())

    # ---------------------------------------------------------------------
    # 4) Tidy up column names so downstream code sees what it expects
    # ---------------------------------------------------------------------
    ds = (
        ds.rename_column("image", "decoded_image")   # PIL.Image
          .rename_column("question",   "question_prompt") # str
          .rename_column("ground_truth", "answer")        # int
    )

    ds = ds.add_column("category", ds["answer"])

    return ds
