import re
from hybrid_inference import uncertainty_aware_hybrid_inference

beverage_ground_truth_list = [
    ["Let me have gin tonic", "1. Place Cup 2. Pour Ice 3. Pour Gin 4. Pour Tonic Water 5. Pour Lemon 6. Serve Beverage 7. Done"],
    ["Cafe mocha, please.", "1. Place Cup 2. Pour Espresso 3. Pour Chocolate 4. Pour Milk 5. Serve Beverage 6. Done"], 
]

def string_to_array(s):
    items = re.split(r'\d+\.\s', s)
    return [item.rstrip() for item in items if item]

for i in beverage_ground_truth_list:
    generated_text, tsr, tr, num_token, time_elapsed = uncertainty_aware_hybrid_inference(i[0], uncertainty_threshold=0.5)
    generated_array = string_to_array(generated_text)

    num_correct_subtask = 0
    for subtask in generated_array:
        if subtask in i[1]:
            num_correct_subtask = num_correct_subtask + 1
    
    task_correct_rate = num_correct_subtask/len(i[1])

    print(f"Generated Plan: \"{generated_text}\", Task Correct Rate: {task_correct_rate}")
    print(f"True Skip Ratio: {tsr}, Transmission Rate: {tr}")
    print(f"Num Tokens: {num_token}, End-to-End Inference Latency: {time_elapsed}s, Token Throughput: {num_token/time_elapsed}")