import torch
import os
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model  # Import LoRA-related functions

# Set CUDA allocation configuration before any CUDA-related imports
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:24"

dataset = load_dataset("json", data_files="test.json")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

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
    output_dir="./tinyllama-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
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

trainer.save_model("./tinyllama-finetuned")
tokenizer.save_pretrained("./tinyllama-finetuned")
