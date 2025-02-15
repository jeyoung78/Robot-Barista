'''
import torch
from transformers import AutoModelForCausalLM
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepSeek-VL2')))

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-tiny"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

## single image conversation example
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n<|ref|>Explain the image to me.<|/ref|>.",
        "images": ["./images/image.png"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
# run the model to get the response
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)
'''

import torch
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
import sys
import os

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
        # Use a set for fast lookup.
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # scores: (batch_size, vocab_size)
        # Create a new scores tensor that is -inf for disallowed tokens.
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
for num in range(1, 257):
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
        "content": "<image>\n<|ref|>I have specified that you only output 8 tokens. Output 1 to 8 in order<|/ref|>.",
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
    max_new_tokens=8,
    min_new_tokens=8,
    logits_processor=logits_processor,
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
# answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
# print(f"{prepare_inputs['sft_format'][0]}", answer)
