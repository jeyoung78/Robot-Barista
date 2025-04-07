import torch
import requests  # Used for making the HTTP request to the server
from slm import slm_inference
# Note: The local llm_verification import is now removed because we call it remotely.
from transformers import AutoTokenizer

model_dir = "./tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

def detokenize(token_id):
    if isinstance(token_id, int):
        return tokenizer.decode(token_id)
    else:
        return tokenizer.decode(token_id[0], skip_special_tokens=True)

# Returns the final generated prompt and the true skip ratio.
def uncertainty_aware_hybrid_inference(prompt: str, max_new_tokens: int = 100, uncertainty_threshold: float = 0.5):
    resample = 0
    num_inference = 0
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token} "
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    for i in range(max_new_tokens):
        num_inference = i
        draft_token_id, uncertainty, draft_distribution = slm_inference(generated=generated)
        
        # Prepare payload for remote llm_verification.
        if isinstance(draft_distribution, torch.Tensor):
            draft_distribution_list = draft_distribution.tolist()
        else:
            draft_distribution_list = draft_distribution
        
        payload = {
            "draft_distribution": draft_distribution_list,
            "draft_token_id": int(draft_token_id),
            "generated": generated.squeeze(0).tolist()
        }
        
        # Replace "165.132.40.52" with the actual IP address of your server.
        server_url = "http://165.132.40.52:5000/llm_verification"
        try:
            response = requests.post(server_url, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            final_token_id = data['result_token_id']
            accepted = data['accepted']
            if not accepted:
                resample += 1
        except Exception as e:
            print("Error calling remote llm_verification:", e)
            final_token_id = draft_token_id

        chosen_token = torch.tensor([final_token_id], device=device)
        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)
        if detokenize(chosen_token) == "Done":
            break

    tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
    return detokenize(generated)[initial_length:], tsr

if __name__ == "__main__":
    prompt_text = "Can I have vanilla latte?"
    generated_text, tsr = uncertainty_aware_hybrid_inference(prompt_text)
    print("Generated text:", generated_text)
    print("True skip ratio:", tsr)
