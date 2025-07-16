from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
import torch
from typing import List, Dict, Any
from PIL import Image
from torch import amp
#supported models: HF: OpenGVLab/InternVL3-8B-hf
#supported models: vLLM: OpenGVLab/InternVL3-9B (unverified)

class InternVL:
    """
    Wrapper for InternVL‑Chat / InternVL3 family.

    Parameters expected in `args`
    --------------------------------
    • gen_engine          : "hf" | "vllm"
    • gen_prompt_suffix   : str
    • max_new_tokens      : int
    • top_k, top_p        : int | float
    • temperature         : float   (0 ⇒ greedy)
    • n_generations       : int
    """

    def __init__(self, args, model_name_path: str="OpenGVLab/InternVL3-8B-hf"):
        # ─── common ──────────────────────────────────────────────────────────
        self.processor = AutoProcessor.from_pretrained(model_name_path, trust_remote_code=True)
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine        = args.gen_engine.lower()
        self.max_new_tokens    = args.max_new_tokens
        self.top_k             = args.top_k
        self.top_p             = args.top_p
        self.temperature       = args.temperature
        self.n_generations     = args.n_generations
        self.do_sample         = self.temperature > 0

        # ─── model loading ──────────────────────────────────────────────────
        if self.gen_engine == "hf":
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name_path,
                torch_dtype="auto",
                device_map="auto"
            ).eval()

            tok = self.processor.tokenizer
            if tok.pad_token_id is None:            
                tok.pad_token = tok.eos_token       

            self.gen_params = dict(
                num_return_sequences=self.n_generations,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=tok.pad_token_id,      
            )

            if self.do_sample:                    
                self.gen_params.update(
                    do_sample=True,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature,
                )
                
        elif self.gen_engine == "vllm":
            self.model = LLM(
                model=model_name_path,
                dtype="auto",
                limit_mm_per_prompt={"image": 1},
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True
            )

            self.gen_params = SamplingParams(
                n=self.n_generations,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown gen_engine: {self.gen_engine}")

    # ─────────────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _make_convs(self, batch: List[Dict[str, Any]]):
        convs = []
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            img    = item["decoded_image"]        # PIL.Image
            convs.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text",  "text": prompt},
                        ],
                    }
                ]
            )
        return convs

    def _make_convs_vllm(self, batch: List[Dict[str, Any]]):
        convs = []
        image_tok = "<image>"  # InternVL3-HF uses literal <image> tokens
        for item in batch:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            convs.append(
                [
                    {
                        "role": "user",
                        "content": f"{image_tok}\n{prompt}",
                    }
                ]
            )
        return convs

    # ─── HF generation ──────────────────────────────────────────────────────
    def _gen_hf(self, batch):
        convs      = self._make_convs(batch)
        text_prompts = self.processor.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )
        images = [c[0]["content"][0]["image"] for c in convs]

        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # generate
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
            gen_ids = self.model.generate(**inputs, **self.gen_params)

        seq_len = inputs.input_ids.size(1)
        trimmed = [
            gen_ids[i * self.n_generations + j][seq_len:]
            for i in range(inputs.input_ids.size(0))
            for j in range(self.n_generations)
        ]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # ─── vLLM generation ────────────────────────────────────────────────────
    def _gen_vllm(self, batch):
        convs        = self._make_convs_vllm(batch)
        text_prompts = self.processor.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=True
        )
        images = [item["decoded_image"] for item in batch]

        requests = [
            {"prompt": p, "multi_modal_data": {"image": img}}
            for p, img in zip(text_prompts, images)
        ]
        outputs = self.model.generate(requests, self.gen_params)
        return [seq.text for out in outputs for seq in out.outputs]


    # ─── public API ─────────────────────────────────────────────────────────
    def generate_response(self, batch: List[Dict[str, Any]]):
        if self.gen_engine == "hf":
            return self._gen_hf(batch)
        elif self.gen_engine == "vllm":
            return self._gen_vllm(batch)

    def shutdown(self):
        if self.gen_engine == "vllm":
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
            except ImportError:
                from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
            
            destroy_model_parallel()

            if hasattr(self.model, "shutdown"):
                self.model.shutdown()

        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()