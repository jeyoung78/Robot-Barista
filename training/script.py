'''
import csv
import torch
import sys
import os
import glob
import numpy as np
import pandas as pd
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

root_dir = 'images/traj_group0'
count = 0
# For each directory that starts with "traj" under root_dir
for i in range(0, 100):
    traj_dir = os.path.join(root_dir, f"traj{i}/images0")
    results = []

    if os.path.isdir(traj_dir):
        for j in range(0, 1):
            img_file = glob.glob(os.path.join(traj_dir, f"im_{j}.jpg"))
            # print(img_file)
            if not img_file:
                # No files found, skip this iteration
                print("no matching")
                continue
            count = count + 1
            print(count)
            
            img_file = img_file[0]

            conversation = [
                {
                    "role": "<|User|>",
                    "content": "<image>\nTo grab the green spatula which way should the robot arm move? Right or Left?",
                    "images": [img_file],
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

            # Get the vocabulary, which is a dictionary mapping tokens to their IDs.
            vocab = tokenizer.get_vocab()

            # Print tokens sorted by their token ID:
            for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                # print(f"{token_id}: {token}")
                pass

            outputs = vl_gpt.language.generate(
                inputs_embeds=inputs_embeds,
                # attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                max_new_tokens=512,
                min_new_tokens=1,
                # logits_processor=logits_processor,
                do_sample=False,
                use_cache=True
            )
            
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            results.append([img_file, answer])
            print(img_file, answer)
            

    # Convert the list of dictionaries to a DataFrame
    # df = pd.DataFrame(results)
    # Write the DataFrame to an Excel file
    # df.to_excel(f'exp_data/results_{i}.xlsx', index=False)

print("Complete!")
'''

import torch
from transformers import AutoModelForCausalLM
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DeepSeek-VL2')))

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images

class RoboticsTransformer:
    def __init__(self):
        # specify the path to the model
        self.MODEL_PATH = "deepseek-ai/deepseek-vl2-tiny"
        self.vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(self.MODEL_PATH)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(self.MODEL_PATH, trust_remote_code=True)
        self.vl_gpt = self.vl_gpt.to(torch.bfloat16).cuda().eval()

    def policy_generate(self):
        ## single image conversation example
        # "content": "<image>\n<|ref|>Describe the image<|/ref|>.",
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image>\nThe relative location (right or left) of the white and red cup to the center of the image is",
                "images": ["saved.jpg"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        print(f"{prepare_inputs['sft_format'][0]}", answer)

rt = RoboticsTransformer()
rt.policy_generate()