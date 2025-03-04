# Goal now: move robot to grab a cup and move to a different location
# Monday: Implement robot code and Communicate class so that robot can be controlled from python script
import pyttsx3
import time
import PIL.Image

from google import genai
from image_processing import ImageProcessing, CameraInterface
from control import Communicate
from llm_highlevel import RecipeGeneration

client = genai.Client(api_key="")

def main():
    # query = "Human: I want you to bring me the rice chips from the drawer? Robot: To do this, the first thing I would do is to\n"
    
    # rg = RecipeGeneration()
    # rg.generate("caramel macchiato")
    '''
    query = "Human: I spilled my coke, can you bring me something to clean it up? Robot: To do this, the first thing I would do is to\n"
    policy_generation = LLMScoring()
    scores = policy_generation.local_llm_scoring(query, options=policy_generation.options, option_start="\n", verbose=False)
    print(scores)
    
    # rt = RoboticsTransformer()
    # rt.policy_generate()
    '''

    url_save = 'saved.jpg'
    ip = ImageProcessing(url = url_save)
    ci = CameraInterface(url = url_save)
    co = Communicate()
    cx, cy = 0, 0
    # 550 <= cx <= 600 and 
    while not (450 <= cy <= 500):
        ci.capture_iamge()
        cx, cy = ip.detect_red_dot()
        print(cx, cy)
        
        if cy > 500:
            co.move_y(False)
            print('move y neg')

        if cy < 450:
            co.move_y(True)
            print('move y pos')
        

    while not (475 <= cx <= 525):
        ci.capture_iamge()
        cx, cy = ip.detect_red_dot()
        print(cx, cy)
        
        if cx > 525:
            co.move_x(False)
            print('move x neg')

        if cx < 475:
            co.move_x(True)
            print('move x pos')

    rg = RecipeGeneration("whiskey sour")
    ingredients = rg.generate()

    target_word = "proceed"

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    
    for ingredient in ingredients:
        print(ingredient)
        engine.say("pour " + ingredient)
        engine.runAndWait()
        co.prepare(True)
        
        while True:
            user_input = input("Type a word: ").strip()
            if user_input.lower() == target_word.lower():
                print("Proceeding...")
                break
            else:
                pass

        co.communicate("pour")
        time.sleep(15)
    
    print("complete!")
    engine.say("Complete!")
    engine.runAndWait()

if __name__ == "__main__":
    main() 

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

# 오늘: 논문 읽기, SayCan Planning 초기형, finetuning 준비. 