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
    def __init__(self, model_path="./gpt2_recipe_generation"):
        """
        Initializes Scoring by loading the fine-tuned GPT2 model and tokenizer.
        """
        self.model_path = model_path
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define the allowed options for generation
        self.options = ["Esp", "Ste", "Milk", "Car", "Sy", "Done"]
        self.full_options = ["Espresso", "Steamed Milk", "Milk", "Caramel Syrup", "Done"]
        self.allowed_token_ids = [self.tokenizer.encode(opt, add_prefix_space=True)[0] for opt in self.options]

    def generate(self, prompt_text):
        prompt_text = f"<recipe_generation> {prompt_text} <recipe_generation> 1. "
        generated_sequence = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        updated_prompt = prompt_text
        toggle = True
        curr = ""
        count = 2

        while "Done" not in updated_prompt:
            if toggle:
                # Generate logits for the next token and restrict to allowed tokens.
                outputs = self.model(generated_sequence)
                next_token_logits = outputs.logits[:, -1, :]
                filtered_logits = next_token_logits.clone()
                
                # Mask all tokens except allowed tokens.
                mask = torch.ones_like(filtered_logits, dtype=torch.bool)
                for token_id in self.allowed_token_ids:
                    mask[:, token_id] = False
                filtered_logits[mask] = -float("Inf")
                
                # If all candidates are masked, fallback to original logits.
                if torch.all(torch.isinf(filtered_logits)):
                    selected_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                else:
                    selected_token_id = torch.argmax(filtered_logits, dim=-1).unsqueeze(0)
                    
                generated_sequence = torch.cat((generated_sequence, selected_token_id.to(self.device)), dim=1)
                updated_prompt = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                print(updated_prompt)
                print(f"Chosen Token: {self.tokenizer.decode(selected_token_id[0])}")
                print("-" * 50)
                toggle = False
                
            else:
                outputs = self.model(generated_sequence)
                next_token_logits = outputs.logits[:, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)
                topk_values, topk_indices = torch.topk(next_token_logits, k=6, dim=-1)
                topk_tokens = [self.tokenizer.decode(idx.item()) for idx in topk_indices[0]]
                
                selected_token_id = topk_indices[0][0].unsqueeze(0).unsqueeze(0)
                curr += self.tokenizer.decode(selected_token_id.squeeze())
                generated_sequence = torch.cat((generated_sequence, selected_token_id.to(self.device)), dim=1)
                updated_prompt = self.tokenizer.decode(generated_sequence[0], skip_special_tokens=True)
                
                print(updated_prompt)
                print(f"Chosen Token: {self.tokenizer.decode(selected_token_id[0])}")
                print("-" * 50)
                
                time.sleep(0.2)
                if curr in self.full_options:
                    # word_to_add = "Ġ{count}.Ġ"

                    # Encode the word to get the token IDs (this may return more than one token)
                    # word_ids = self.tokenizer.encode(word_to_add, return_tensors="pt")

                    # Optionally, ensure word_ids is on the same device as generated_sequence:
                    # word_ids = word_ids.to(generated_sequence.device)
                    # count = count + 1
                    curr = ""
                    toggle = True
        
        pattern = r'\d+\.\s*([A-Za-z ]+)'

        # Extract all matches.
        matches = re.findall(pattern, updated_prompt)

        # Clean up the matches by stripping whitespace and converting to lowercase.
        array = [match.strip().lower() for match in matches]

        return updated_prompt, array

# Example usage:
if __name__ == "__main__":
    scoring = Scoring()
    final_prompt = scoring.generate("Can I have Caramel Macchiato?")
    print("Final updated prompt:")
    print(final_prompt)
