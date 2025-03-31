import openai
import ast
import re
import tiktoken
import torch

import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

class Scoring:
    def __init__(self, model_path="./gpt2_recipe"):
        """
        Initializes Scoring by loading the fine-tuned GPT2 model and tokenizer.
        """
        self.model_path = model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define the allowed options for generation
        self.options = ['Place', 'Cup', 'Pour', 'Water', 'Ice', 'Esp', 'resso', 'Serve', "Bever", 'age', 'Done']
        self.full_options = ['PlaceCup', "PourWater", "PourIce", "PourEspresso", "ServeBeverage", "Done"]
        self.allowed_token_ids = [self.tokenizer.encode(opt, add_prefix_space=True)[0] for opt in self.options]
        print(self.allowed_token_ids)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def top_k_top_p_filtering(self, logits, top_k=50, top_p=0.95):
        values, _ = torch.topk(logits, top_k)
        kth_value = values[:, -1].unsqueeze(1)
        logits = torch.where(logits < kth_value, torch.full_like(logits, float('-inf')), logits)

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
        return logits
    
    def allowed_tokens_filtering(self, logits, allowed_token_ids):
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[:, allowed_token_ids] = True
        # Set logits for all tokens not allowed to -infinity.
        filtered_logits = logits.masked_fill(~mask, -float('inf'))
        return filtered_logits


    def generate(self, prompt_text):
        prompt_text = f"<recipe_generation> {prompt_text} <recipe_generation> 1."
        generated_sequence = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        past_key_calues = None 
        count = 1
        curr = ''
        
        while True:
            outputs = self.model(generated_sequence)
            next_token_logits = outputs.logits[:, -1, :]
            special_token_id = self.tokenizer.convert_tokens_to_ids("<recipe_generation>")
            next_token_logits[:, special_token_id] = -float('inf')

            topk_values, topk_indices = torch.topk(next_token_logits, k=6, dim=-1)
            topk_tokens = [self.tokenizer.decode(idx.item()) for idx in topk_indices[0]]
            index = 0

            for topk_token in topk_tokens:
                if topk_token.strip() in self.options:
                    selected_token_id = topk_indices[0, index]
                    selected_token_id = selected_token_id.unsqueeze(0).unsqueeze(0)

                    selected_token_id = selected_token_id.to(generated_sequence.device)
                    generated_sequence = torch.cat((generated_sequence, selected_token_id), dim=1)
                    updated_prompt = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                    # print(updated_prompt.replace("Ġ", " "))
                    
                    if topk_token.strip() == 'Done':
                        return 

                    curr += topk_token.strip()
                    print(f"curr: {topk_token}")
                    if curr.strip() in self.full_options:   
                        count += 1
                        formatted_str = f"Ġ{count}."
                        encoded_ids = self.tokenizer.encode(formatted_str, add_special_tokens=False)
                        token_tensor = torch.tensor(encoded_ids, dtype=torch.long).unsqueeze(0).to(generated_sequence.device)
                        generated_sequence = torch.cat((generated_sequence, token_tensor), dim=1)
                        curr = ""
                        
                    break

                index = index + 1

            
            time.sleep(0.25)

if __name__ == "__main__":
    scoring = Scoring()
    final_prompt = scoring.generate("I want iced Americano.")
    print("Final updated prompt:")
    print(final_prompt)
