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