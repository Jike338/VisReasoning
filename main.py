import argparse
import json

from tqdm import tqdm
from PIL import Image as PILImage
from argparse import Namespace
from datasets import load_dataset, Dataset
from utils import save_response_to_json 
from load_clevr_counting import load_jsonl_with_images_superclever_counting
from load_mix_data import load_json_mixed_data
from load_tom import load_jsonl_hi_tom, load_jsonl_explore_tom, load_jsonl_fantom, load_jsonl_tom_mix
from load_vl_data import load_vis_reasoning

def get_task_name(args):
    if "mathvista" in args.dataset_name_path.lower():
        args.task_name = "mathvista"
    if all(x in args.dataset_name_path.lower() for x in ("superclevr", "counting")):
        args.task_name = "superclevr_counting"
    if "mix_data" in args.dataset_name_path.lower():
        args.task_name = "mix_data"
    if "hi-tom" in args.dataset_name_path.lower():
        args.task_name = "hi_tom"
    if "exploretom" in args.dataset_name_path.lower():
        args.task_name = "explore_tom"
    if "fantom" in args.dataset_name_path.lower():
        args.task_name = "fantom"
    if "tom" in args.dataset_name_path.lower():
        args.task_name = "tom_mix"
    if "visreasoning" in args.dataset_name_path.lower():
        args.task_name = "visreasoning"

def get_data_list(task_name, dataset_name_path, dataset_split, dataset_dir, sample, seed):

    if "mathvista" in task_name:
        data_list = load_dataset(dataset_name_path, split=dataset_split)

    if "superclevr_counting" in task_name:
        data_list = load_jsonl_with_images_superclever_counting(dataset_name_path, dataset_dir)

    if "mix_data" in task_name:
        data_list = load_json_mixed_data(json_path=dataset_name_path)

    if "hi_tom" in task_name:
        data_list = load_jsonl_hi_tom(jsonl_path=dataset_name_path)

    if "explore_tom" in task_name:
        data_list = load_jsonl_explore_tom(jsonl_path=dataset_name_path)
    
    if "fantom" in task_name:
        data_list = load_jsonl_fantom(jsonl_path=dataset_name_path)

    if "tom_mix" in task_name:
        data_list = load_jsonl_tom_mix(jsonl_path=dataset_name_path)
    
    if "visreasoning" in task_name:
        data_list = load_vis_reasoning(dataset_name_path, split=dataset_split)
    # ------------------------------------------------------------------
    # Optional sampling logic (unchanged)
    # ------------------------------------------------------------------
    if sample > 0:
        if isinstance(data_list, Dataset):
            data_list = data_list.shuffle(seed=seed).select(range(sample))
        else:  # Should not reach here any more
            raise TypeError(f"Unexpected data_list type {type(data_list)}")

    return data_list

def format_datalist(task_name, data_list):
    # must have keys: question_prompt, decoded_image 
    if "mathvista" in task_name.lower():
        data_list = data_list.rename_column("query", "question_prompt")
        
        # Qwen2VL’s patch size
        MIN_SIDE = 28
        MAX_PIXELS = 250_000

        def _resize(example):
            img: PILImage = example["decoded_image"]
            w, h = img.size

            # 1) down-scale if too many pixels
            total = w * h
            if total > MAX_PIXELS:
                scale = (MAX_PIXELS / total) ** 0.5
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), PILImage.LANCZOS)
                w, h = img.size

            # 2) up-sample if either side is below the minimum
            if w < MIN_SIDE or h < MIN_SIDE:
                new_w = max(w, MIN_SIDE)
                new_h = max(h, MIN_SIDE)
                img = img.resize((new_w, new_h), PILImage.LANCZOS)

            example["decoded_image"] = img
            return example

        data_list = data_list.map(_resize)

    return data_list

def get_gen_prompt_suffix(gen_prompt_suffix_type):
    prompt_suffix = ""
    if gen_prompt_suffix_type == "dir":
        prompt_suffix = "\n Output the answer directly, without anything else."
    elif gen_prompt_suffix_type == "cot":
        prompt_suffix = "\n First output your thinking process first and then output the answer at the end."
    elif gen_prompt_suffix_type == "cot_tag":
        prompt_suffix = "\n First output the thinking process in <think> </think> and then final answer (number) in <answer> </answer> tags."
    else:
        prompt_suffix = gen_prompt_suffix_type #custom
    return prompt_suffix

def get_model(model_name_path, args):
    model = None
    if all(k in model_name_path.lower() for k in ("qwen", "vl")):
        from models import qwenvl
        model = qwenvl.QwenVL(args, model_name_path)
    elif "qwen" in model_name_path.lower():
        from models import qwen
        model = qwen.Qwen(args, model_name_path)
    elif "llava" in model_name_path.lower():
        from models import llava
        model = llava.LLaVA(args, model_name_path)
    elif "internvl" in model_name_path.lower():
        from models import internvl
        model = internvl.InternVL(args, model_name_path)
    elif "gpt" in model_name_path.lower() or "o3" in model_name_path.lower() or "o4" in model_name_path.lower():
        from models import gpt
        model = gpt.GPT(model_name_path, args)
    return model

def generate_raw_responses(args):
    args.duty_type = "raw"
    get_task_name(args)

    # get data 
    data_list = get_data_list(args.task_name, args.dataset_name_path, args.dataset_split, args.dataset_dir, args.sample, args.seed)
    reformatted_data_list = format_datalist(args.task_name, data_list)

    # get model
    args.gen_prompt_suffix = get_gen_prompt_suffix(args.gen_prompt_suffix_type)
    model = get_model(args.model_name_path, args)
    
    try:
        # generation loop
        item_with_raw_response = []
        for batch_idx in tqdm(range(0, len(reformatted_data_list), args.bs)):
            end_idx = min(batch_idx + args.bs, len(reformatted_data_list))
            batch_inputs = [reformatted_data_list[i] for i in range(batch_idx, end_idx)]
            batch_outputs = model.generate_response(batch_inputs)
            print(batch_outputs)
            # update each example
            for idx, question in enumerate(batch_inputs):
                question = {k: v for k, v in question.items() if k != "decoded_image"}
                question["raw_response"] = batch_outputs[idx*args.n_generations:(idx+1)*args.n_generations]
                item_with_raw_response.append(question)
    finally:
        if model and hasattr(model, "shutdown"):
            model.shutdown()
        del model
        import torch
        torch.cuda.empty_cache()
    
    save_response_to_json(args, item_with_raw_response)

def extract_answer_from_raw_responses(args, use_vanilla_extract, use_quick_extract_w_gpt, use_gpt_extract, use_answer_tag_extract):
    with open(args.file_with_raw_response, "r", encoding="utf-8") as f:
        data_with_raw_responses = json.load(f)

    saved_config_dict = data_with_raw_responses["parameters"]
    for k, v in saved_config_dict.items():
        setattr(args, k, v)
    
    from extract import Extractor
    answer_extractor = Extractor(
        data_with_raw_responses['result'],
        args,
        use_vanilla_extract=use_vanilla_extract,
        use_quick_extract_w_gpt=use_quick_extract_w_gpt,
        use_gpt_extract=use_gpt_extract,
        use_answer_tag_extract=use_answer_tag_extract,
    )

    answer_extractor.extract_ans_and_save()

def calc_score_from_extracted_responses(args):
    with open(args.file_with_extracted_response, "r", encoding="utf-8") as f:
        data_with_extracted_responses = json.load(f)

    saved_config_dict = data_with_extracted_responses["parameters"]
    for k, v in saved_config_dict.items():
        setattr(args, k, v)

    from score import Scorer
    answer_scorer = Scorer(data_with_extracted_responses['result'], args)
    answer_scorer.calc_score_and_save()

def parse_arguments():
    parser = argparse.ArgumentParser()

    # for generation
    parser.add_argument("--duty_generation", action="store_true", default=False)
    parser.add_argument("--sample", default=-1, type=int)
    parser.add_argument("--seed", default=2025, type=int)

    # for extraction
    parser.add_argument("--duty_extract_answer", action="store_true", default=False)
    parser.add_argument("--file_with_raw_response", default=None, type=str)
    parser.add_argument("--use_vanilla_extract", action="store_true", default=False)
    parser.add_argument("--use_quick_extract_w_gpt", action="store_true", default=False)
    parser.add_argument("--use_gpt_extract", action="store_true", default=False)
    parser.add_argument("--use_answer_tag_extract", action="store_true", default=False)

    # for scoring
    parser.add_argument("--duty_calc_score", action="store_true", default=False)
    parser.add_argument("--file_with_extracted_response", default=None, type=str)

    # generic args
    parser.add_argument("--dataset_name_path", default=None, type=str) # AI4Math/MathVista
    parser.add_argument("--dataset_split", default=None, type=str) # testmini
    parser.add_argument("--dataset_dir", default=None, type=str) # "/scratch1/jikezhon/R1-V/src/eval/images"
    parser.add_argument("--outputs_dir", default="./results", type=str) # ./results
    parser.add_argument("--tag", default=None, type=str) # dir_hf
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--delete_prev_file", action="store_true", default=False)

    # model args
    parser.add_argument('--bs', default=8, type=int) 
    parser.add_argument('--model_name_path', default="Qwen/Qwen2-VL-2B-Instruct", type=str) 
    parser.add_argument("--gen_prompt_suffix_type", default="", type=str)
    parser.add_argument("--gen_engine", type=str, default="hf", choices=["hf", "openai", "vllm"])
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Temperature for sampling")
    parser.add_argument("--n_generations", type=int, default=1,
                        help="Number of independent generations per input")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    if args.duty_generation:
        assert args.dataset_name_path is not None
        assert args.model_name_path is not None
        generate_raw_responses(args)
    
    if args.duty_extract_answer:
        assert args.file_with_raw_response is not None
        extract_answer_from_raw_responses(args, args.use_vanilla_extract, args.use_quick_extract_w_gpt, args.use_gpt_extract, args.use_answer_tag_extract)
    
    if args.duty_calc_score:
        assert args.file_with_extracted_response is not None
        calc_score_from_extracted_responses(args)
    
# Notes, to add a new dataset, modify the following functions
# get_task_name()
# get_data_list()
# format_datalist()
# _get_normalized_preds() in Score()
    
     