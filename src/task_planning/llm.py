
'''
import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens):
    banned_words = ["in", "into", "In"]
    banned_token_ids = []
    for word in banned_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            banned_token_ids.extend(tokens)

    accepted = True

    prefix_text = "Generate a unique robot recipe as a numbered list using only these actions: Place, Pour, Serve, and Done. Each step must include one action followed by a single ingredient name—no amounts or extra descriptions. Do not include any additional words like into, in, or extra descriptors. Follow this exact format: 1. Place Cup 2. Pour Water 3. Pour Espresso 4. Serve Beverage 5. Done. Here's actual order: "
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(generated.device)
    generated = torch.cat((prefix_tokens, generated), dim=1)

    with torch.no_grad():
        outputs = model(generated)
    
    logits = outputs.logits  
    next_token_logits = logits[0, -1, :]
    for token_id in banned_token_ids:
        next_token_logits[token_id] = -float('Inf')

    allowed_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
    allowed_token_ids = torch.tensor(allowed_tokens, device=next_token_logits.device)
    allowed_mask[allowed_token_ids] = True

    disallowed_value = float('-inf')
    masked_logits = torch.where(allowed_mask, next_token_logits, torch.tensor(disallowed_value, device=next_token_logits.device))

    target_distribution = torch.softmax(masked_logits, dim=-1)
    
    if not isinstance(draft_distribution, np.ndarray):
        draft_distribution = np.array(draft_distribution)

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

app = Flask(__name__)

@app.route('/llm_verification', methods=['POST'])
def call_llm_verification():
    data = request.get_json()
    draft_distribution = data['draft_distribution']
    draft_token_id = data['draft_token_id']
    generated_list = data['generated']
    allowed_tokens = data['allowed_tokens']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert the received list back into a tensor; assuming generated is a list of token ids.
    generated = torch.tensor([generated_list], device=device)
    result_token_id, accepted = llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens)
    return jsonify({'result_token_id': result_token_id, 'accepted': accepted})

if __name__ == '__main__':
    print("Starting llm.py server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
'''

import numpy as np
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import difflib

app = Flask(__name__)

# Load the pre-trained tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load cafe recipes from the JSON file (ensure label.json is in the same directory)
with open('label.json', 'r') as f:
    cafe_recipes = json.load(f)

def llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens):
    # --- Perform Recipe Lookup Using Fuzzy Matching ---
    # Decode the provided generated tokens (a list of token IDs) into text.
    # We assume the beginning of this text represents the drink name.
    candidate_lookup = tokenizer.decode(generated[0].tolist()).strip().split("\n")[0]
    
    lookup_recipe_text = None
    if candidate_lookup:
        # Build a list of available drink names
        drink_names = [recipe['prompt'] for recipe in cafe_recipes]
        # Find the best match (highest similarity) for the candidate lookup text
        matches = difflib.get_close_matches(candidate_lookup, drink_names, n=1, cutoff=0.4)
        if matches:
            # Retrieve the recipe associated with the best matching drink name
            matching_recipe = next((r for r in cafe_recipes if r['prompt'] == matches[0]), None)
            if matching_recipe:
                lookup_recipe_text = matching_recipe['response']
    
    # --- Build the Prefix for Generation ---
    if lookup_recipe_text:
        # Include the lookup result into the prefix if a match was found.
        prefix_text = (
            f"Based on the recommended recipe for '{candidate_lookup}': {lookup_recipe_text} "
            "Generate a unique robot recipe as a numbered list using only these actions: Place, Pour, Serve, and Done. "
            "Each step must include one action followed by a single ingredient name—no amounts or extra descriptions. "
            "Do not include any additional words like into, in, or extra descriptors. Follow this exact format: "
            "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Serve Beverage 5. Done. Here's actual order: "
        )
    else:
        # If no match was found, fall back to a default instruction.
        prefix_text = (
            "Generate a unique robot recipe as a numbered list using only these actions: Place, Pour, Serve, and Done. "
            "Each step must include one action followed by a single ingredient name—no amounts or extra descriptions. "
            "Do not include any additional words like into, in, or extra descriptors. Follow this exact format: "
            "1. Place Cup 2. Pour Water 3. Pour Espresso 4. Serve Beverage 5. Done. Here's actual order: "
        )
    
    # Tokenize the prefix and prepend it to the generated tokens.
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt")["input_ids"].to(generated.device)
    generated = torch.cat((prefix_tokens, generated), dim=1)
    
    # --- LLM Verification (as before) ---
    banned_words = ["in", "into", "In"]
    banned_token_ids = []
    for word in banned_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        if tokens:
            banned_token_ids.extend(tokens)

    with torch.no_grad():
        outputs = model(generated)
    
    logits = outputs.logits  
    next_token_logits = logits[0, -1, :]
    for token_id in banned_token_ids:
        next_token_logits[token_id] = -float('Inf')
    
    allowed_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
    allowed_token_ids = torch.tensor(allowed_tokens, device=next_token_logits.device)
    allowed_mask[allowed_token_ids] = True
    
    disallowed_value = float('-inf')
    masked_logits = torch.where(allowed_mask, next_token_logits,
                                torch.tensor(disallowed_value, device=next_token_logits.device))
    
    target_distribution = torch.softmax(masked_logits, dim=-1)
    
    # Ensure the draft distribution is a numpy array
    if not isinstance(draft_distribution, np.ndarray):
        draft_distribution = np.array(draft_distribution)
    
    target_distribution = target_distribution.cpu().numpy()
    x_d = draft_distribution[draft_token_id]
    y_d = target_distribution[draft_token_id]
    
    if x_d <= y_d or np.random.rand() < y_d / x_d:
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
    
    return result_token_id, accepted

@app.route('/llm_verification', methods=['POST'])
def call_llm_verification():
    data = request.get_json()
    draft_distribution = data['draft_distribution']
    draft_token_id = data['draft_token_id']
    generated_list = data['generated']
    allowed_tokens = data['allowed_tokens']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Convert the received list of token IDs back into a tensor.
    generated = torch.tensor([generated_list], device=device)
    result_token_id, accepted = llm_verification(draft_distribution, draft_token_id, generated, allowed_tokens)
    return jsonify({'result_token_id': result_token_id, 'accepted': accepted})

if __name__ == '__main__':
    print("Starting llm.py server on port 5000...")
    app.run(host='0.0.0.0', port=5000)
