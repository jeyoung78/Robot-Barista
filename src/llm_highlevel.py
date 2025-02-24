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
        prompt = f"""
            You are a robotic assistant for a cafe that prepares a wide variety of drinks.
            Your task is to generate a detailed action plan—a sequence of actions—to fulfill the user’s high-level drink-making request.
            Your response MUST be formatted as a numbered list only. Each step should start with a number followed by a period (e.g., "1.") and sentences MUST start with put. Do not include any additional text or commentary outside the list.
            For example:
            Instruction: Make a cappuccino.
            Response:
            1. Prepare cup
            2. Pour espresso
            3. Add milk
            4. Stir
            5. Serve drink
            6. Done
            Now, for this command: "{instruction}", generate the action plan in the exact numbered list format described above.
            """
        # Tokenize the input text
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Generate output (adjust parameters as needed)
        output_ids = self.model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=300
            # max_new_tokens=512,
            # do_sample=False,
        )

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(generated_text)

        return generated_text

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