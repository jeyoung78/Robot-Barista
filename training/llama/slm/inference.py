import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the directory where your fine-tuned model and tokenizer are saved.
model_dir = "./tinyllama-finetuned"

# Load the tokenizer from your fine-tuned directory.
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load the fine-tuned model with ignore_mismatched_sizes=True to allow for the updated vocabulary size.
model = AutoModelForCausalLM.from_pretrained(model_dir, ignore_mismatched_sizes=True)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Define your special token and prepare your prompt.
special_token = "<recipe_generation>"
prompt_text = "Hi, can I get a Blackberry Ginger Latte please"
input_text = f"{special_token} {prompt_text} {special_token}"

# Tokenize the input text.
inputs = tokenizer(input_text, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate the response with desired generation parameters.
with torch.no_grad():
    generated_ids = model.generate(
        inputs["input_ids"],
        max_length=128,         # Adjust max length as needed.
        do_sample=True,         # Enable sampling for more varied outputs.
        temperature=0.7,        # Adjust temperature for randomness.
        top_p=0.9,              # Use nucleus sampling.
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode and print the output. skip_special_tokens=True removes special tokens from the output.
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Output:", output_text)
