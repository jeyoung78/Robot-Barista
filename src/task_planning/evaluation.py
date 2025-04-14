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


# Iterate over each entry in the JSON list
for entry in beverage_ground_truth_list:
    prompt = entry["prompt"]
    response = entry["response"]
    
    # Generate the plan using the hybrid inference function
    generated_text, tsr, tr, num_token, time_elapsed = uncertainty_aware_hybrid_inference(prompt, uncertainty_threshold=0.05)
    
    # Convert both generated text and the ground truth response into arrays of steps
    generated_array = string_to_array(generated_text)
    ground_truth_array = string_to_array(response)
    
    num_correct_subtask = 0
    for subtask in generated_array:
        # Here we check if the generated subtask exists in the ground truth response
        if subtask in response.lower():
            num_correct_subtask += 1
    
    # Calculate the task correct rate; if ground_truth_array is empty, avoid division by zero
    task_correct_rate = num_correct_subtask / len(ground_truth_array) if ground_truth_array else 0


    print(f"Task Correct Rate: {task_correct_rate}")
    print(f"Prompt: {prompt}")
    print(f"Generated Plan:{generated_text}")
    print(f"Ground Truth: {response}")
    print(f"True Skip Ratio: {tsr}, Transmission Rate: {tr}")
    print(f"Num Tokens: {num_token}, End-to-End Inference Latency: {time_elapsed}s, Token Throughput: {num_token/time_elapsed}")
    print("-"*20)
