import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

class LLMScoring:
    def __init__(self):
        # Load your local model; you can change this to another supported model.
        self.MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # or "EleutherAI/gpt-neo-125M", "decapoda-research/llama-7b-hf", etc.
        # self.MODEL_NAME = "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, 
            trust_remote_code=True
        )
        self.model.eval()

        self.options = [
            " Go to drawer.",
            " Open the drawer.",
            " Take the rice chips out of the drawer.",
            " Close the drawer.",
            " Pick up the rice chip.",
            " Bring it to you.", 
            " Put down the rice chips.",
            " Done." 
        ]   

    def score_prompt(self, query: str, option: str, option_start: str="\n", verbose: bool=False):
        """
        Computes the total log probability for the option tokens appended to the query.
        The summing stops if a token matching option_start is encountered.
        
        Args:
            query: The initial prompt text.
            option: The option text that is appended to the prompt.
            option_start: A token (or string) indicating where to stop scoring.
            verbose: Whether to print token-by-token info.
        
        Returns:
            total_log_prob: The summed log probability for the option tokens.
            tokens: The list of tokens (as strings) for the prompt.
            token_log_probs: The list of log probabilities (floats) corresponding to each token.
        """
        prompt_options = query + option
        # print(prompt_options)       
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        inputs = self.tokenizer(prompt_options, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids
        
        with torch.no_grad():
            outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Compute log probabilities from logits
        log_probs = F.log_softmax(logits, dim=-1)  # shape: [1, seq_len, vocab_size]
        
        # Convert token IDs to tokens (strings)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        query_tokens = self.tokenizer(query).input_ids
        option_start_index = len(query_tokens)

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
    
    def local_llm_scoring(self, query: str, options: list, option_start: str="\n", verbose: bool=False):
        """
        Scores a list of options appended to the query by computing their total log probability.
        
        Args:
            query: The base query string.
            options: A list of option strings to score.
            option_start: Token/string at which scoring stops.
            verbose: Whether to print detailed information.
        
        Returns:
            scores: A dict mapping each option to its total log probability.
        """
        scores = {}
        for option in options:
            score, tokens, token_log_probs = self.score_prompt(query, option, option_start, verbose)
            scores[option] = score
            if verbose:
                print(f"Option: {option}\nTotal Log Probability: {score:.4f}\n{'-'*40}")
        return scores
    
    def download_model(self):
        model_name = "deepseek-ai/DeepSeek-V3"
        downloaded_files = snapshot_download(repo_id=model_name, revision="v3")

        print("Downloaded files are located at:", downloaded_files)