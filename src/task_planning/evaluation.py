import json
import re
from hybrid_inference import uncertainty_aware_hybrid_inference
import random

# Load the JSON file with 500 beverage entries
with open("test.json", "r") as f:
    beverage_ground_truth_list = json.load(f)

def string_to_array(s):
    items = re.split(r'\d+\.\s', s)
    return [item.rstrip() for item in items if item]

random.shuffle(beverage_ground_truth_list)

scores = []

# Iterate over each entry in the JSON list
for entry in beverage_ground_truth_list:
    prompt = entry["prompt"]
    response = entry["response"]
    
    generated_text, tsr, tr, num_token, time_elapsed = uncertainty_aware_hybrid_inference(prompt, uncertainty_threshold=0.25)
    
    generated_array = string_to_array(generated_text)
    ground_truth_array = string_to_array(response)
    length = len(ground_truth_array)
    
    num_correct_subtask = 0
   

    for i, subtask in enumerate(generated_array):
        for idx, gt_step in enumerate(ground_truth_array):
            if subtask.lower().strip() == gt_step.lower().strip():
                num_correct_subtask = num_correct_subtask + 1
                ground_truth_array.pop(idx)
    
    task_correct_rate = num_correct_subtask / length 
    scores.append(task_correct_rate)

    print(f"Task Correct Rate: {task_correct_rate}")
    print(f"Prompt: {prompt}")
    print(f"Generated Plan:{generated_text}")
    print(f"Ground Truth: {response}")
    print(f"True Skip Ratio: {tsr}, Transmission Rate: {tr}")
    print(f"Num Tokens: {num_token}, End-to-End Inference Latency: {time_elapsed}s, Token Throughput: {num_token/time_elapsed}")
    print(f"Score Average: {sum(scores)/len(scores)}")
    print("-"*50)
