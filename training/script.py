'''
import torch
import sys
import os
import numpy as np

from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepSeek-VL2')))

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

# Specify the path to the model and load the processor and tokenizer.
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

# Load the model and prepare it for inference.
vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

class ActionTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]
        return mask

def discretize(value, min_val, max_val, num_bins=256): 
    value = np.clip(value, min_val, max_val) 
    bin_width = (max_val - min_val) / num_bins 
    bin_index = int((value - min_val) / bin_width) 
    if bin_index >= num_bins: 
        bin_index = num_bins - 1 
    return bin_index + 1 

allowed_token_ids = set()
allowed_token_ids.add(tokenizer.eos_token_id)

for num in range(1, 256):
    token_str = str(num)
    # Tokenize without adding special tokens.
    token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
    print(token_str, token_ids)
    if len(token_ids) == 1:
        allowed_token_ids.add(token_ids[0])
    else:
        print(f"Warning: The number {num} does not map to a single token.")

# Create a LogitsProcessorList with our allowed-token processor.
logits_processor = LogitsProcessorList([ActionTokensLogitsProcessor(allowed_token_ids)])

conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\nI have specified that you only output 8 tokens. Output 1 to 8 in order.",
        "images": ["./images/image.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# Load images and prepare inputs.
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# Run image encoder to get the image embeddings.
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    max_new_tokens=512,
    # min_new_tokens=8,
    # logits_processor=logits_processor,
    do_sample=False,
    use_cache=True
)

# Extract the generated token IDs from the sequence.
generated_ids = outputs[0].cpu().tolist()

# Iterate and print each token.
for idx, token_id in enumerate(generated_ids):
    token_str = tokenizer.decode([token_id], skip_special_tokens=True)
    print(f"Token {idx}: {token_str} (ID: {token_id})")

# Decode and print the answer.
answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
'''
#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
    get_linear_schedule_with_warmup,
)

# Append the DeepSeek-VL2 module folder to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepSeek-VL2')))

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


class ActionTokensLogitsProcessor(LogitsProcessor):
    """
    Logits processor that allows only specific token IDs (e.g. digits as action tokens).
    """
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]
        return mask


def discretize(value, min_val, max_val, num_bins=256): 
    value = np.clip(value, min_val, max_val) 
    bin_width = (max_val - min_val) / num_bins 
    bin_index = int((value - min_val) / bin_width) 
    if bin_index >= num_bins: 
        bin_index = num_bins - 1 
    return bin_index + 1 


def build_allowed_token_ids(tokenizer):
    allowed_token_ids = set()
    # Always allow EOS.
    allowed_token_ids.add(tokenizer.eos_token_id)

    # Allow tokens corresponding to numbers 1 through 255.
    for num in range(1, 256):
        token_str = str(num)
        # Tokenize without adding special tokens.
        token_ids = tokenizer(token_str, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            allowed_token_ids.add(token_ids[0])
        else:
            print(f"Warning: The number {num} did not map to a single token.")
    return allowed_token_ids


# ------------------------------
# Placeholder Dataset Definition
# ------------------------------
class RT2Dataset(Dataset):
    """
    A placeholder dataset. Expects each item to be a dict with:
      - "conversation": a list of conversation turns (each turn is a dict)
      - "labels": a list of token ids representing the expected action tokens
    """
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        conversation = item["conversation"]
        # Load images (if any) using the provided helper.
        pil_images = load_pil_images(conversation)
        # Process the conversation and images.
        inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        )
        # For training, we assume the processor returns input_ids and attention_mask.
        # Also assume item["labels"] is already tokenized (a list of ints).
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        labels = item["labels"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


# ------------------------------
# Training Function
# ------------------------------
def train(args):
    model_path = "deepseek-ai/deepseek-vl2-tiny"
    print(f"Loading model and processor from {model_path}...")
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda()
    model.train()

    # --------------
    # Prepare Dataset
    # --------------
    # Replace this with your actual data loading mechanism.
    # Each data sample must include:
    #    - "conversation": a list of conversation turns with keys "role", "content", and optional "images"
    #    - "labels": the expected target token ids (list of ints)

    dummy_conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nI have specified that you only output 8 tokens. Output 1 to 8 in order.",
            "images": ["./images/image.png"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    # For demonstration, we assume the target is the tokenized form of "1 2 3 4 5 6 7 8"
    target_text = "1 2 3 4 5 6 7 8"
    target_ids = tokenizer(target_text, add_special_tokens=False).input_ids

    # Create a dummy dataset with one example.
    data = [{"conversation": dummy_conversation, "labels": target_ids}]
    dataset = RT2Dataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --------------
    # Optimizer & Scheduler
    # --------------
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # --------------
    # Training Loop
    # --------------
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            # Move batch tensors to GPU.
            batch = {k: v.cuda() for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            loss.backward()

            # optimizer.step()
            # scheduler.step()
            # optimizer.zero_grad()

            if step % args.log_interval == 0:
                print(f"Epoch {epoch+1}/{args.epochs} Step {step}: Loss = {loss.item():.4f}")

    # Save the fine-tuned model.
    save_path = args.output_dir 
    print(f"Saving fine-tuned model to {save_path}...")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    

# ------------------------------
# Inference Function
# ------------------------------
def inference(args):
    model_path = args.model_path or "deepseek-ai/deepseek-vl2-tiny"
    print(f"Loading model and processor from {model_path} for inference...")
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()

    # Build allowed token IDs for the logits processor.
    allowed_token_ids = build_allowed_token_ids(tokenizer)
    logits_processor = LogitsProcessorList([ActionTokensLogitsProcessor(allowed_token_ids)])

    # Example conversation: prompt with an image.
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\nI have specified that you only output 8 tokens. Output 1 to 8 in order.",
            "images": [args.image_path or "./images/image.png"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    # Process images and conversation.
    pil_images = load_pil_images(conversation)
    inputs = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt=""
    ).to(model.device)

    # Run the image encoder (if applicable) to get inputs_embeds.
    inputs_embeds = model.prepare_inputs_embeds(**inputs)

    # Generate tokens with constraints.
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
        logits_processor=logits_processor,
    )

    # Extract generated token IDs.
    generated_ids = outputs[0].cpu().tolist()
    print("Generated Action Tokens:")
    for idx, token_id in enumerate(generated_ids):
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)
        print(f"Token {idx}: {token_str} (ID: {token_id})")

    # Decode and print the full answer.
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("Full decoded answer:")
    print(answer)


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek-VL2 to generate action tokens from a prompt and image, "
                    "and/or run inference with a constrained generation."
    )
    parser.add_argument("--mode", choices=["train", "inference"], required=True, help="Mode: train or inference")
    parser.add_argument("--batch_size", type=int, default=, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=1, help="Steps between logging loss during training")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model for inference")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image used in inference")
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "inference":
        inference(args)


if __name__ == "__main__":
    main()

