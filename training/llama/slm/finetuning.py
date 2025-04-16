import os

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model

# Load both JSON files
data1 = load_dataset("json", data_files="data_collection.json")["train"]
data2 = load_dataset("json", data_files="test.json")["train"]

# Ensure 1:1 ratio by truncating the longer dataset
min_len = min(len(data1), len(data2))
data1 = data1.select(range(min_len))
data2 = data2.select(range(min_len))

# Interleave the datasets
combined_data = Dataset.from_dict({
    "prompt": data1["prompt"] + data2["prompt"],
    "response": data1["response"] + data2["response"]
})

# Shuffle and split into train/test
dataset = combined_data.train_test_split(test_size=0.1, seed=42)
first_example = dataset['train'][0]

# Print the actual content of the 'prompt' and 'response'
# print("Prompt:", first_example['prompt'])
# print("Response:", first_example['response'])

model_path = "meta-llama/Llama-2-7b-chat-hf"
model_path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # or torch.bfloat16 if supported
    bnb_4bit_quant_type="nf4",               # "nf4" is often used for a better balance
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,  # Pass the quantization configuration
    device_map={"": "cuda:2"}
)

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    # Option 1: Use the EOS token as the pad token.
    tokenizer.pad_token = tokenizer.eos_token


special_token = "<recipe_generation>"

lora_config = LoraConfig(
    r=8,                  
    lora_alpha=32,        
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.05,    
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

MAX_LENGTH = 128

def preprocess_function(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for prompt_text, response_text in zip(examples["prompt"], examples["response"]):
        prompt_part = f"{special_token} {prompt_text} {special_token}"
        full_text = prompt_part + " " + response_text

        tokenized = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        tokenized_prompt = tokenizer(prompt_part, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
        prompt_len = len(tokenized_prompt["input_ids"])

        labels = input_ids.copy()
        for i in range(prompt_len):
            labels[i] = -100

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

tokenized_dataset = dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=dataset["train"].column_names
)

training_args = TrainingArguments(
    output_dir="./llm-finetuned",
    num_train_epochs=4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    fp16=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

trainer.train()

trainer.save_model("./llm-finetuned")
tokenizer.save_pretrained("./llm-finetuned")
