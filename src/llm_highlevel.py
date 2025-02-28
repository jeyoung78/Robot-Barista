import google.generativeai as genai
import PIL.Image
import json
import ast

class RecipeGeneration():
    def __init__(self, beverage):
        self.api_key = ""
        genai.configure(api_key=self.api_key)

        self.beverage = beverage
        self.prompt = f"""
        Provide a valid Python list containing only liquid ingredients and ice for this beverage, to be poured: {self.beverage}. 
        Skip those that are not liquid or ice. 
        If syrup is to go in, add the word syrup. So if vanilla syrup has to go in, for example, one element within an array should "vanilla_syrup"
        Do not include solid ingredients, amounts, measurements, or explanations. 
        Strictly output a valid Python list with ingredient names as strings.
        Nothing more than a Python array.
        Order to ingredients in an order that it needs to be poured. 
        """

        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')

    def generate(self):
        try:
            response = self.model.generate_content(self.prompt)  
            response.resolve() 

            words = response.text.split()
            print(words)

        except Exception as e:
            print(f"An error occurred: {e}")

        # 1. Filter out the code-fence lines
        filtered_lines = [line for line in words if not line.startswith("```")]

        # 2. Join the remaining lines into a single string
        raw_string = "".join(filtered_lines)

        # 3. Safely parse the string as a Python literal
        my_array = ast.literal_eval(raw_string)

        print(my_array)

        return my_array

rg = RecipeGeneration("caramel macchiato")
ingredients = rg.generate()
for ingredient in ingredients:
    print(ingredient)
'''
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

class RecipeGeneration:
    def __init__(self):
        # Load your local model; you can change this to another supported model.
        # self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or "EleutherAI/gpt-neo-125M", "decapoda-research/llama-7b-hf", etc.
        self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, 
            trust_remote_code=True
        )
        self.model.eval()

    def generate(self, instruction):
        prompt = f"""Provide a valid Python list containing only liquid, ice, or powder ingredients for this beverage: {instruction}.
        Do not include solid ingredients, amounts, measurements, or explanations. Strictly output a **valid Python list** with ingredient names as strings.
        Example: ["milk", "coffee", "ice", "vanilla syrup"]"""
        # Tokenize the input text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate output (adjust parameters as needed)
        output_ids = self.model.generate(
            input_ids=input_ids,
            max_length=150,
            temperature=0.2,
            top_p=1.0,
            top_k=1
            # max_new_tokens=512,
            # do_sample=False,
        )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        print(generated_text)

        return generated_text
'''
'''
class LLMScoring:
    def __init__(self):
        # Load your local model; you can change this to another supported model.
        # self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or "EleutherAI/gpt-neo-125M", "decapoda-research/llama-7b-hf", etc.
        self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, 
            trust_remote_code=True
        )
        self.model.eval()
        self.options = option_list = [
            # "prepare cup",
            " pour espresso",
            " pour tea",
            " pour hot chocolate",
            " add ice",
            " add vanilla syrup",
            " add caramel",
            # " add milk",
            " add cinnamon",
            " stir",
            " serve drink",
            " done"
        ]        

    def score_prompt(self, user_instruction: str, option: str, option_start: str="\n", verbose: bool=False):
        prompt = f"""
            You are a robotic assistant for a cafe that prepares a wide variety of drinks using a set of predefined actions. Your task is to generate a detailed plan—a sequence of actions—that fulfills the user’s high-level drink-making request.

            Examples 1 
            Instruction: Make a cappucino. Response: 1. prepare cup 2. pour espresso 3. add milk 4. stir 5. serve drink 6. done
            Examples 2
            Instruction: Prepare an iced latte. Response: 1. prepare cup 2. pour espresso 3. add ice 4. add vanilla syrup 5. add milk 6. stir 7. serve drink 8. done
            Be careful with order of actions. 
            
            Now, generate an action plan for the following instruction: {user_instruction} I would 1. prepare cup 2. add milk 3. a: 
            """
        prompt_options = prompt + '\n' + option
        # print(prompt_options)       
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(prompt_options, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids

        device = torch.device("cuda")
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Compute log probabilities from logits
        log_probs = F.log_softmax(logits, dim=-1)  # shape: [1, seq_len, vocab_size]
        
        # Convert token IDs to tokens (strings)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        query_tokens = self.tokenizer(prompt).input_ids
        option_start_index = len(query_tokens) - 1

        total_log_prob = 0.0
        token_log_probs = []

        for i in range(option_start_index, input_ids.size(1)):
            if i == 0:
                continue
            token_id = input_ids[0, i].item()
            token_str = tokens[i]
            if token_str == option_start:
                if verbose:
                    print(f"Encountered termination token '{option_start}' at position {i}.")
                break

            # Log probability of token i is found at position i-1
            token_log_prob = log_probs[0, i-1, token_id].item()
            token_log_probs.append(token_log_prob)
            total_log_prob += token_log_prob
            
            if verbose:
                print(f"Token: {token_str}\tTokenid: {token_id:.4f}\tLogProb: {token_log_prob:.4f}\tProb: {math.exp(token_log_prob):.4f}")
        
        return total_log_prob, tokens, token_log_probs
    
    def local_llm_scoring(self, user_instruction: str, option_start: str="\n", verbose: bool=False):
        scores = {}
        options = self.options
        for option in options:
            score, tokens, token_log_probs = self.score_prompt(user_instruction, option, option_start, verbose)
            scores[option] = score
            if verbose:
                print(f"Option: {option}\nTotal Log Probability: {score:.4f}\n{'-'*40}")
        return scores
'''