import openai
import tiktoken
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 전역 캐시 (동일 요청에 대해 재사용)
LLM_CACHE = {}
ENGINE = "EleutherAI/gpt-neo-125M"

# OpenAI API 키 설정 (반드시 본인의 API 키로 교체)

class Scoring:
    def __init__(self):
        self.MODEL_NAME = ENGINE
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True
        )
        self.model.eval()
        self.options = [
        "espresso.",
        "milk.",
        "ice.",
        "vanilla syrup.",
        "water.",
        "caramel syrup.",
        "done."
        ]

    def score_prompt(self, query: str, option: str, option_start: str="\n", verbose: bool=False):
        prompt_options = query + option
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(prompt_options, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        query_tokens = self.tokenizer(query).input_ids
        option_start_index = len(query_tokens)

        total_log_prob = 0.0
        token_log_probs = []

        for i in range(option_start_index, input_ids.size(1)):
            if i == 0:
                continue
            token_id = input_ids[0, i].item()
            token_str = tokens[i]
            if token_str == option_start:
                if verbose:
                    print(f"Encountered termination token '{option_start}' at position {i}.")
                break
            token_log_prob = log_probs[0, i-1, token_id].item()
            token_log_probs.append(token_log_prob)
            total_log_prob += token_log_prob

        return total_log_prob, tokens, token_log_probs

    def local_llm_scoring(self, query: str, options: list, option_start: str="\n", verbose: bool=False):
        scores = {}
        for option in options:
            score, tokens, token_log_probs = self.score_prompt(query, option, option_start, verbose)
            scores[option] = score
        return scores

def normalize_scores(scores):
    max_score = max(scores.values()) if scores else 1
    normed_scores = {key: np.clip(scores[key] / max_score, 0, 1) for key in scores}
    return normed_scores

def generate_prompt(beverage: str, score_threshold: float=-5.0, verbose: bool=False) -> str:
    base_prompt = f"""
Robot: Hi there, I'm a robot operating in an office cafe.
Robot: You can ask me to make various drinks and I'll tell you the sequence of actions I would take to prepare them in order.

Human: How would you make Cappuccino?
Robot: 1. espresso, 2. milk, 3. ice, 4. done.

Human: How would you make Vanilla Latte?
Robot: 1. vanilla syrup, 2. espresso, 3. milk, 4. ice,  5. done.

Human: How would you make Americano?
Robot: 1. ice, 2. water, 3. espresso, 4. done.

Human: How would you make a Caramel Macchiato?
Robot: 1. vanilla syrup, 2. milk, 3. espresso, 4. ice, 5. caramel syrup 6. done.

Human: How would you make {beverage}?
Robot: """

    scorer = Scoring()
    remaining_options = scorer.options.copy()
    chosen_steps = []
    current_prompt = base_prompt
    while remaining_options:
        scores = scorer.local_llm_scoring(current_prompt, remaining_options, option_start="\n", verbose=verbose)
        if not scores:
            break
        best_option = max(scores, key=scores.get)
        best_score = scores[best_option]
        if best_score < score_threshold:
            print(f"Option '{best_option.strip()}' has score {best_score:.4f} which is below threshold. Appending 'Done.'")
            chosen_steps.append("Done.")
            break
        print(f"Option '{best_option.strip()}' added with score: {best_score:.4f}")
        chosen_steps.append(best_option.strip())
        current_prompt += best_option
        remaining_options.remove(best_option)
    numbered_steps = [f"{i+1}. {step}" for i, step in enumerate(chosen_steps)]
    final_prompt = base_prompt + " " + " ".join(numbered_steps)
    return final_prompt

def main():
    beverage = "americano"
    score_threshold = -25.0
    final_prompt = generate_prompt(beverage, score_threshold=score_threshold, verbose=True)
    print("\nFinal prompt:")
    print(final_prompt)

if __name__ == "__main__":
    main()