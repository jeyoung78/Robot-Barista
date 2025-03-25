import os
import json
import random
import time

from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset


from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(api_key="AIzaSyAbZpHttVawCw_I-K68XQgHPlKQZ4XXSQg")

prompt = """
The goal: Produce a user request for a coffee order in a cafe. The request can be:
1. A specific drink order (e.g., “I want caramel macchiato”).
2. A modified drink order (e.g., “Can I have a latte with extra shot?”).
3. A vague order (e.g., “I’m not sure what I want”).

Example 1
The user request scenario is 'Specific Drink Order':
Output: I want caramel macchiato.

Example 2
The user request scenario is 'Specific Drink Order':
Output: Could I have a iced latte?

Example 3
The user request scenario is 'Specific Drink Order':
Output: Give me a cappuccino.

Example 4
The user request scenario is 'Modified Drink Order':
Output: I'd like a mocha with half the syrup.

Example 5
The user request scenario is 'Modified Drink Order':
Output: Can I have a latte with an extra shot?

Example 6
The user request scenario is 'Vague Order':
Output: I'm not sure what to get. Maybe something sweet?

Example 7
The user request scenario is 'Vague Order':
Output: I need something very caffetinated.

You do not need to keep the format in the examples. As long as it is a customer ordering drinks, it's good. Return only the output. Don't include flat white. Choose menu as if you're in cafe. 
"""

def generate_single_prompt():
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=[prompt])
    words = response.text.split()
    sentence = " ".join(words)
    print(sentence)
    time.sleep(2)

for i in range(0, 100):
    generate_single_prompt()

'''
def generate_teacher_response(prompt):
    pass

num_pairs = 500
distillation_data = []

for i in range(num_pairs):
    print(f"Processing pair {i+1}/{num_pairs}...")
    prompt = generate_single_prompt()
    teacher_output = generate_teacher_response(prompt)
    # Format as a conversation for the training example
    combined_text = f"<recipe_generation> {prompt}: {teacher_output}"
    print(combined_text)
    distillation_data.append({"text": combined_text})

with open("distillation_dataset_iterative.json", "w") as f:
    json.dump(distillation_data, f, indent=4)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
special_token = "<recipe_generation>"

# Add the special token to the tokenizer's vocabulary
num_added_tokens = tokenizer.add_tokens([special_token])
print(f"Added {num_added_tokens} token(s): {special_token}")

# Resize the model's token embeddings to accommodate the new token
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=128
    )

dataset = Dataset.from_list(distillation_data)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="./gpt2_distilled",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./gpt2_distilled")
tokenizer.save_pretrained("./gpt2_distilled")
'''