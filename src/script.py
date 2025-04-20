import json
from transformers import AutoTokenizer

# Load your Llama tokenizer (replace 'llama-model-name' with your model's name/path)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

def get_unique_tokens(json_file_path):
    # Open and load the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Use a set to collect unique token IDs
    token_set = set()
    
    # Choose which fields to use; here we process both "prompt" and "response"
    for entry in data:
        for field in ["prompt", "response"]:
            text = entry.get(field, "")
            # Tokenize the text without adding extra special tokens
            tokens = tokenizer.encode(text, add_special_tokens=False)
            token_set.update(tokens)
    
    # Convert the set to a sorted list if needed (sorted is optional)
    allowed_tokens = sorted(token_set)
    return allowed_tokens

# Example usage; ensure that "drinks.json" is the correct file path
allowed_tokens = get_unique_tokens("mega_coffee_data/modified_order_recipe.json")
print("Unique allowed tokens:", allowed_tokens)
for token_id in allowed_tokens:
    token_text = tokenizer.decode([token_id])
    # print(f"Token ID: {token_id}  ->  Text: '{token_text}'")