from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# 1. Load the fine-tuned model and tokenizer
model_path = "./gpt2_distilled"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 2. Prepare your prompt
#    If you used a special token (e.g., <recipe_generation>) during training, 
#    make sure to include it here as part of your prompt format.
prompt_text = "<recipe_generation> Can I get an iced caramel macchiato with extra espresso shot? <recipe_generation>"

# 3. Tokenize the prompt
input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

# 4. Generate a response
#    Adjust parameters like max_length, temperature, etc., to control output.
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.7,    # Lower temperature = more deterministic
        top_p=0.9,          # Nucleus sampling
        do_sample=True
    )

# 5. Decode and print the model's output
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("Model Output:", output_text)
