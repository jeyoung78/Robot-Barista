import os
import json
import random
import time

from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset


from google import genai
from google.genai.types import HttpOptions, Part

client = genai.Client(api_key="AIzaSyAbZpHttVawCw_I-K68XQgHPlKQZ4XXSQg")

def generate_single_prompt():
    prompt_order = """
        The goal: Produce a user request for a coffee order in a cafe. The request can be:
        1. A specific drink order (e.g., “I want caramel macchiato”).
        2. A modified drink order (e.g., “Can I have a latte with extra shot?”).

        Example 1
        The user request scenario is 'Specific Drink Order':
        Output: I want caramel macchiato.

        Example 2
        The user request scenario is 'Specific Drink Order':
        Output: Could I have a iced latte?

        Example 5
        The user request scenario is 'Modified Drink Order':
        Output: Can I have a latte with an extra shot of espresso?

        Example 5
        The user request scenario is 'Modified Drink Order':
        Output: Can I have a caramel macchiato with an extra shot of espresso?

        Example 6
        The user request scenario is 'Modified Drink Order':
        Output: Can I have a iced caramel macchiato with an extra caramel?

        Example 8
        The user request scenario is 'Modified Drink Order':
        Output: Can I have a iced americano with an extra shot of espresso?

        You do not need to keep the format in the examples. As long as it is a customer ordering drinks, it's good. Return only the output. Don't include flat white, regular coffee, brewed coffee, vanilla latte, or hot chocolate. 
        Order only americano. 
        Do not modify drink by replacing milk with oatmilk or soy milk. 
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=[prompt_order])
    words = response.text.split()
    sentence = " ".join(words)
    time.sleep(5)

    return sentence

def generate_teacher_response(prompt):
    prompt_list = f"""
        The goal: Given a user’s coffee order (either specific, modified, or vague), produce a numbered list of ingredients. End with a “done.” 
        Use reasoning to identify the correct recipe, but do not include that reasoning in the final answer—only output the numbered list.

        ---
        Example 1
        User says:
        > I want caramel macchiato

        A:
        - Identify it as a specific drink order.
        - Standard caramel macchiato recipe might include espresso, steamed milk, caramel syrup.
        - Final output is a numbered list of these ingredients.

        Final Output: 1. Espresso 2. Steamed Milk 3. Caramel Syrup 4. Done

        ---
        Example 2
        User says:
        > Can I have a latte with an extra shot?

        A:
        - Recognize it as a modified drink order: base latte plus an extra shot of espresso.
        - The recipe is standard latte ingredients plus an additional espresso shot.

        Final Output: 1. Espresso 2. Espresso 3. Steamed Milk 4. Done
        ---
        Example 5
        User says:
        > Can I have a iced caramel macchiato with an extra caramel?

        A:
        - Recognize it as a modified drink order: base caramel macchiato plus an extra caramel.
        - The recipe is standard caramel macchiato ingredients plus an additional caramel.

        Final Output:
        1. 1. Espresso 2. Steamed Milk 3. Caramel Syrup 4. 3. Caramel Syrup 5. Done

        Now, here's the actual order: {prompt}
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash-thinking-exp-01-21",
        contents=[prompt_list])
    words = response.text.split()
    sentence = " ".join(words)
    time.sleep(5)
    return sentence

num_pairs = 30
distillation_data = []

# Load existing data if file exists
dataset_filename = "data.json"
if os.path.exists(dataset_filename):
    with open(dataset_filename, "r") as f:
        distillation_data = json.load(f)
else:
    distillation_data = []

for i in range(num_pairs):
    print(f"Processing pair {i+1}/{num_pairs}...")
    prompt = generate_single_prompt()
    teacher_output = generate_teacher_response(prompt)
    # Append new data
    distillation_data.append({
        "prompt": prompt,
        "response": teacher_output
    })
    # Write the updated data after each new pair (or you could do it outside the loop)
    with open(dataset_filename, "w") as f:
        json.dump(distillation_data, f, indent=4)