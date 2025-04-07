import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def llm_verification(draft_distribution, draft_token_id, generated):
    accepted = True

    prefix_text = "Generate a unique robot recipe as a numbered list using only these actions: Place, Pour, Serve, and Done. Each step must include one action followed by a single ingredient name—no amounts or extra descriptions. Do not include any additional words like into, in, or extra descriptors. Follow this exact format: 1. Place Cup 2. Pour Water 3. Pour Espresso 4. Serve Beverage 5. Done. Here's actual order: "
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(generated.device)
    generated = torch.cat((prefix_tokens, generated), dim=1)


    with torch.no_grad():
        outputs = model(generated)
    
    logits = outputs.logits  
    next_token_logits = logits[0, -1, :]
    target_distribution = torch.softmax(next_token_logits, dim=-1)

    draft_distribution = draft_distribution.cpu().numpy()
    target_distribution = target_distribution.cpu().numpy()

    x_d = draft_distribution[draft_token_id]
    y_d = target_distribution[draft_token_id]
    # print(type(draft_distribution))

    if x_d <= y_d or np.random.rand() < y_d / x_d:
        acceptance_prob = 1.0
        accepted = True
        result_token_id = draft_token_id
    else:
        diff = np.maximum(target_distribution - draft_distribution, 0)
        total_diff = np.sum(diff)
        if total_diff == 0:
            result_token_id = draft_token_id  
        else:
            normalized_diff = diff / total_diff
            result_token_id = np.random.choice(len(draft_distribution), p=normalized_diff)
            accepted = False

    # print(f"Target: {tokenizer.decode(draft_token_id).strip()}")
    return result_token_id, accepted