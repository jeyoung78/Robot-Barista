import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained("./gpt2_recipe_generation")
model = GPT2LMHeadModel.from_pretrained("./gpt2_recipe_generation")
model.eval()

# Define your input prompt.
prompt = ""
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Generate text while returning scores.
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    output_scores=True,
    return_dict_in_generate=True
)

# Extract the generated token IDs.
generated_ids = output.sequences[0]
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
print("Generated Text:", generated_text)

# Iterate over each step's score and the corresponding token.
for step, (logits, token_id) in enumerate(zip(output.scores, generated_ids[1:]), start=1):
    # Convert logits to probabilities.
    probabilities = torch.softmax(logits, dim=-1)
    # Determine the index of the chosen token from the distribution.
    # Note: For sampling, this is just an approximation; the chosen token is recorded in `generated_ids`.
    chosen_token_idx = torch.argmax(probabilities, dim=-1)
    
    # Decode the tokens.
    chosen_token_from_dist = tokenizer.decode(chosen_token_idx)
    actual_chosen_token = tokenizer.decode(token_id.unsqueeze(0))
    
    print(f"Step {step} probability distribution (non-zero entries shown):")
    non_zero_indices = (probabilities > 0).nonzero(as_tuple=True)[1]
    for idx in non_zero_indices:
        print(f"  Token: {tokenizer.decode(idx.unsqueeze(0))}, Probability: {probabilities[0, idx].item():.4f}")
    
    print(f"Step {step} token chosen from distribution (argmax): {chosen_token_from_dist}")
    print(f"Step {step} actual chosen token: {actual_chosen_token}\n")
