# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

input_text = "We must be the great arsenal of democracy"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate text with a maximum length of 100 tokens.
outputs = model.generate(input_ids, max_length=100)

# Decode the generated tokens.
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
