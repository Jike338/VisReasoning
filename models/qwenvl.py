from vllm import LLM, SamplingParams
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Dict, Any
from PIL import Image
from torch import amp
#supported models: HF: llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf
#supported models: vLLM: same as above

class QwenVL:
    def __init__(
        self,
        args,
        model_name_path: str = "Qwen/Qwen2-VL-2B-Instruct"
    ):
        # shared
        self.processor      = AutoProcessor.from_pretrained(model_name_path)
        self.processor.tokenizer.padding_side = "left"
        self.gen_prompt_suffix = args.gen_prompt_suffix
        self.gen_engine     = args.gen_engine.lower()
        self.max_new_tokens = args.max_new_tokens
        self.top_k          = args.top_k
        self.top_p          = args.top_p
        self.temperature    = args.temperature
        self.n_generations  = args.n_generations
        self.do_sample = self.temperature > 0

        if self.gen_engine == "hf":
            if "qwen2-vl" in model_name_path.lower():
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name_path, torch_dtype="auto", device_map="auto"
                )
            elif "qwen2.5-vl" in model_name_path.lower():
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_path, torch_dtype="auto", device_map="auto"
                )
            self.model.eval()

            # gen params
            self.gen_params = dict(
                num_return_sequences=self.n_generations,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
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
                trust_remote_code=True,
                max_model_len=1024
            )

            # self.model = LLM(
            #     model=model_name_path,
            #     dtype="auto",               # keep it explicit
            #     tensor_parallel_size=1,  # shard weights
            #     max_model_len=1024,             # shrink KV cache
            #     gpu_memory_utilization=0.95,    # leave 15 % head‑room
            #     trust_remote_code=True,
            #     enable_prefix_caching=True,
            #     enforce_eager=True,
            # )

            # gen_params
            self.gen_params = SamplingParams(
                n=self.n_generations,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature, #vllm greedy must have temp = 0, when temp=0, top_p and top_k are ignored
                top_p=self.top_p,
                top_k=self.top_k,
            )
        else:
            raise ValueError(f"Unknown gen_engine: {self.gen_engine}")


    def _make_batched_conversations(self, batched_input: List[Dict[str, Any]]):
        convs = []
        for item in batched_input:
            prompt = item["question_prompt"] + self.gen_prompt_suffix
            img    = item["decoded_image"]  # PIL.Image

            conv = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text": prompt},
                    ],
                }
            ]
            convs.append(conv)
        return convs


    def _gen_hugginface(self, batched_input):
        # prep data
        batched_convs      = self._make_batched_conversations(batched_input)
        text_inputs = self.processor.apply_chat_template(batched_convs, 
                                                         tokenizer=False, 
                                                         add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(batched_convs)
        hf_inputs = self.processor(
            text=text_inputs,
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        # generate
        with torch.no_grad(), amp.autocast(device_type="cuda", dtype=next(self.model.parameters()).dtype):
            gen_ids = self.model.generate(**hf_inputs, **self.gen_params)

        # trim off prompt tokens
        batch_size = hf_inputs.input_ids.size(0)
        seq_len    = hf_inputs.input_ids.size(1)
        trimmed = []
        for i in range(batch_size):
            for j in range(self.n_generations):
                idx = i * self.n_generations + j
                trimmed.append(gen_ids[idx][seq_len:])

        # decode
        batched_output = self.processor.batch_decode(trimmed, 
                                                     skip_special_tokens=True, 
                                                     clean_up_tokenization_spaces=False
        )  

        return batched_output
    
    def _gen_vllm(self, batched_input):
        # prep data
        batched_convs       = self._make_batched_conversations(batched_input)
        text_prompts = self.processor.apply_chat_template(batched_convs, 
                                                          tokenizer=False, 
                                                          add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(batched_convs)

        requests = [
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            }
            for prompt, img in zip(text_prompts, img_inputs)
        ]

        # generate
        outputs = self.model.generate(requests, self.gen_params)

        # decode
        batched_output = []
        for out in outputs:
            for seq in out.outputs:
                batched_output.append(seq.text)
        return batched_output


    def generate_response(self, batched_input: List[Dict[str, Any]]):
        if self.gen_engine == "hf":
            return self._gen_hugginface(batched_input)
        elif self.gen_engine == "vllm":
            return self._gen_vllm(batched_input)

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
