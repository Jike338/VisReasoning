from openai import OpenAI
import time
import traceback

class GPT:
    def __init__(self, 
                 model='gpt-4o-mini',
                 temperature=0,
                 max_tokens=1024,
                 n_generations=1,
                 patience=2):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.n_generations = n_generations
        self.max_patience = patience
        self.client = OpenAI(api_key=API_KEY)

    def extract_answer_from_raw_response(self, prompt):
        messages = [
            {"role": "user",
             "content": [{"type": "text", "text": prompt}]}
        ]
        
        extraction = "None"
        patience = self.max_patience

        while patience > 0:
            patience -= 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1
                )
                pred = response.choices[0].message.content.strip()

                if pred:
                    extraction = pred
                    break
                else:
                    print("Empty response from GPT.")

            except Exception as e:
                print(f"Exception during GPT call: {e.__class__.__name__}: {e}")
                traceback.print_exc()
                print(f"Exception during GPT call: {e}")

        return extraction
