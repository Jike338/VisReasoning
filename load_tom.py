#!/usr/bin/env python3
"""
Hi‑ToM CoT benchmark loader
===========================

• Reads the JSON‑lines file you produced with build_benchmark_json.py
• Builds a `question_prompt` that matches the *training* template:

    You are doing a theory‑of‑mind task …
    ***The story***
    …
    ***The question***
    …

    Output the thinking process in <think> </think> and the final answer
    in <answer> </answer> tags.
      • For MCQ, choose *one* of the provided options.
      • For open‑ended, extract a short phrase from the story.

• Keeps the original fields (story / question / answer / …) for scoring.

After this preprocessing **`format_datalist()` no longer needs a Hi‑ToM
branch** – the prompt is ready to feed into any text‑only model wrapper.
"""

from pathlib import Path
from datasets import load_dataset, Dataset


INSTR_HEADER = (
    "You are doing a theory of mind task, where you need to read a story "
    "and answer a question based on the story."
)

# INSTR_FOOTER= ("For multiple-choice questions, strictly select one of the provided choices.\n"
#     "For open-ended questions, the answer is located in the story — please only extract and return a short phrase.")
# INSTR_FOOTER= ("")
# INSTR_FOOTER = (
#     """Output your final answer directly. 
#     \nFor multiple-choice questions, strictly select one of the provided choices.  
#     \nFor open-ended questions, the answer is located in the story, please only extract and return a short phrase."""
# )
INSTR_FOOTER = (
    "Output the thinking process in <think> </think> and final answer "
    "in <answer> </answer> tags.\n"
    "For multiple‑choice questions, strictly select *one* of the provided "
    "choices.\n"
    "For open‑ended questions, the answer is located in the story; please "
    "only extract and return a short phrase inside <answer> </answer>."
)
# INSTR_FOOTER = (
#     """First output your final answer directly in <answer> </answer> tags and then explain your answer in <think> </think>. 
#     \nFor multiple-choice questions, strictly select one of the provided choices.  
#     \nFor open-ended questions, the answer is located in the story, please only extract and return a short phrase."""
# )
# def _build_prompt(story: str, question: str, note: str | None = "") -> str:
#     """Return a single ChatML‑ready user prompt string."""
#     parts = [
#         INSTR_HEADER,
#         "\n***The story***\n",
#         story.strip(),
#         "\n***The question***\n",
#         question.strip(),
#     ]
#     if note:
#         parts.extend(["\n***Additional note***\n", note.strip()])
#     parts.extend(["\n\n", INSTR_FOOTER])
#     return "".join(parts)

# def _build_prompt(story: str, question: str) -> str:
#     return f""" You are doing a theory of mind task, where you need to read a story and answer a question based on the story.
#     ***The story*** 
#     {story} 
#     ***The question*** 
#     {question}

# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.
# \nFor multiple-choice questions, strictly select one of the provided choices.  
# \nFor open-ended questions, the answer is located in the story, please only extract and return a short phrase inside the <answer> </answer> tag. """

# def _build_prompt(story: str, question: str) -> str:
#     return (
#         "You are doing a theory of mind task, where you need to read a story and answer a question based on the story.\n\n"
#         "***The story***\n"
#         f"{story.strip()}\n\n"
#         "***The question***\n"
#         f"{question.strip()}\n\n"
#         "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
#         "For multiple-choice questions, strictly select one of the provided choices.\n"
#         "For open-ended questions, the answer is located in the story — please only extract and return a short phrase."
#     )

# def _build_prompt(story: str, question: str) -> str:
#     return f""" You are doing a theory of mind task, where you need to read a story and answer a question based on the story.
#     ***The story*** 
#     {story} 
#     ***The question*** 
#     {question}
# \nFor multiple-choice questions, strictly select one of the provided choices.  
# \nFor open-ended questions, the answer is located in the story, please only extract and return a short phrase inside the <answer> </answer> tag.
# Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."""
# import re

# def clean_eval_story(story: str) -> str:
#     # 1) Remove line numbers at the start of each line
#     story = re.sub(r'^\s*\d+\s+', '', story, flags=re.MULTILINE)
#     # 2) Replace all newlines with a space
#     story = story.replace('\n', ' ')
#     # 3) Collapse multiple spaces into one (optional, but neat)
#     story = re.sub(r'\s+', ' ', story).strip()
#     return story

# def _build_prompt(story: str, question: str) -> str:
#     # story = clean_eval_story(story)
#     return f""" You are doing a theory of mind task, where you need to read a story and answer a question based on the story. \nNote: You should assume the following.\n(1) An agent witnesses everything and every movement before exiting a room.\n(2) An agent A can infer another agent B's mental state only if A and B have been in the same room, or have private or public interactions.
# \nNow read the following story and answer the question
# ***The story*** 
#     {story} 
# ***The question*** 
#     {question} 
# Output the thinking process in <think> </think> and final answer (a short phrase) in <answer> </answer> tags."""


# def _build_prompt(story: str, question: str) -> str:
#     return (
#         "You are doing a theory of mind task, where you need to read a story and answer a question based on the story.\n\n"
#         "***The story***\n"
#         f"{story}\n\n"
#         "***The question***\n"
#         f"{question}\n\n"
#         "Answer:\n"
#     )

# def _build_prompt(story: str, question: str) -> str:
#     return (
#         "You are doing a theory of mind task, where you need to read a story and answer a question based on the story.\n\n"
#         "***The story***\n"
#         f"{story}\n\n"
#         "***The question***\n"
#         f"{question}\n\n"
#         "Output your final answer directly\n."
#     )

# # ******************************** RFT nothink
# def _build_prompt(story: str, question: str) -> str:
#     return f"""You are a helpful assistant. The assistant answers the question directly without anything else.

# Note: You should assume the following.
# (1) An agent witnesses everything and every movement before exiting a room.
# (2) An agent A can infer another agent B’s mental state only if A and B have been in the same room, or have private or public interactions.

# Read the following story and answer the question.

# Story: {story}
# Question: {question}"""

# ******************************** RFT
def _build_prompt(story: str, question: str) -> str:
    return f"""You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a ToM reasoning problem. After thinking, when you finally reach a conclusion, clearly state your answer within <answer> </answer> tags.

Note: You should assume the following.
(1) An agent witnesses everything and every movement before exiting a room.
(2) An agent A can infer another agent B’s mental state only if A and B have been in the same room, or have private or public interactions.

Read the following story and answer the question.

Story: {story}
Question: {question}"""

# # ******************************** SFT
# def _build_prompt(story: str, question: str) -> str:
#     return f"""Read the following story and answer the question.
# Story: {story}
# Question: {question}"""

def _build_prompt_fantom(story: str, question: str, options: list) -> str:
    return f""" You are doing a theory of mind task, where you need to read a story and answer a question based on the story.
    ***The story*** 
    {story} 
    ***The question*** 
    {question}
    ***The choices***
    {options}
Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."""


def load_jsonl_fantom(jsonl_path: str | Path) -> Dataset:
    """
    Load the FANTOM JSONL file and append a `question_prompt` column.
    Each prompt includes the short_context, the question, and the options.
    """
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    # sanity-check essential columns
    for col in ("short_context", "question", "options"):
        if col not in ds.column_names:
            raise ValueError(f"[FANTOM loader] missing '{col}' in {jsonl_path}")

    def _add_prompt(example):
        example["question_prompt"] = _build_prompt_fantom(
            example["full_context"],
            example["question"],
            example["options"]
        )
        example["answer"] = example.pop("solution")
        return example

    ds = ds.map(_add_prompt, desc="Building FANTOM evaluation prompts", load_from_cache_file=False)
    return ds


def load_jsonl_hi_tom(jsonl_path: str | Path) -> Dataset:
    """Load the JSONL file and append a `question_prompt` column."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    # sanity‑check essential columns
    for col in ("story", "question", "answer"):
        if col not in ds.column_names:
            raise ValueError(f"[Hi‑ToM loader] missing “{col}” in {jsonl_path}")

    def _add_prompt(example):
        example["question_prompt"] = _build_prompt(
            example["story"],
            example["question"],
            # example.get("note", ""),
        )
        return example

    ds = ds.map(_add_prompt, desc="Building Hi‑ToM evaluation prompts", load_from_cache_file=False)
    return ds

def load_jsonl_explore_tom(jsonl_path: str | Path) -> Dataset:
    """Load the JSONL file and append a `question_prompt` column."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    # sanity‑check essential columns
    for col in ("infilled_story", "question", "expected_answer"):
        if col not in ds.column_names:
            raise ValueError(f"[Hi‑ToM loader] missing “{col}” in {jsonl_path}")

    def _add_prompt(example):
        example["question_prompt"] = _build_prompt(
            example["infilled_story"],
            example["question"],
        )
        # Rename "expected_answer" to "answer"
        example["answer"] = example.pop("expected_answer")
        return example

    ds = ds.map(_add_prompt, desc="Building Hi‑ToM evaluation prompts", load_from_cache_file=False)
    return ds

def load_jsonl_tom_mix(jsonl_path: str | Path) -> Dataset:
    """Load the JSONL file and append a `question_prompt` column."""
    ds = load_dataset("json", data_files=str(jsonl_path), split="train")

    # sanity‑check essential columns
    for col in ("story", "question", "data_source"):
        if col not in ds.column_names:
            raise ValueError(f"[Hi‑ToM loader] missing “{col}” in {jsonl_path}")

    def _add_prompt(example):
        example["question_prompt"] = _build_prompt(
            example["story"],
            example["question"],
        )
        # example["question_prompt"] = example["prompt"][0]["content"]
        # Rename "expected_answer" to "answer"
        category = example["data_source"]
        if category.lower() == "hi_tom":
            question_order = example["extra_info"]["question_order"]
            if question_order == 4:
                category = "hi_tom_4"
            else:
                category = "hi_tom_less_4"
        example["category"] = category
        return example

    ds = ds.map(_add_prompt, desc="Building Hi‑ToM evaluation prompts", load_from_cache_file=False)
    return ds