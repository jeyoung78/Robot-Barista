import torch
import time
import requests  # For making HTTP requests to the server
from slm import slm_inference
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
    start = time.time()
    resample = 0
    num_inference = 0
    num_transmission = 0
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token} "
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    for i in range(max_new_tokens):
        num_inference = i + 1
        # Get the draft token, its uncertainty, and distribution from SLM inference.
        draft_token_id, uncertainty, draft_distribution = slm_inference(generated=generated)
        
        # Check uncertainty: if high, use remote llm_verification; otherwise, use the SLM token.
        if uncertainty > uncertainty_threshold:
            num_transmission = num_transmission + 1
            print(f"High uncertainty ({uncertainty:.2f} > {uncertainty_threshold}); calling remote LLM verification...")
            # Convert draft_distribution to a list if needed.
            if isinstance(draft_distribution, torch.Tensor):
                draft_distribution_list = draft_distribution.tolist()
            else:
                draft_distribution_list = draft_distribution
            
            payload = {
                "draft_distribution": draft_distribution_list,
                "draft_token_id": int(draft_token_id),
                "generated": generated.squeeze(0).tolist()
            }
            
            # Replace "165.132.40.52" with your server's actual IP address.
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
        else:
            print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
            final_token_id = draft_token_id

        chosen_token = torch.tensor([final_token_id], device=device)
        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)
        print(detokenize(generated))

        if detokenize(chosen_token).strip() == "Done":
            break

    tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
    tr = (1 - num_transmission/num_inference) if num_inference > 0 else 1.0
    time_elapsed = time.time() - start
    return detokenize(generated)[initial_length:], tsr, tr, num_inference, time_elapsed

if __name__ == "__main__":
    prompt_text = "Can I have caramel macchiato??"
    generated_text, tsr, tr = uncertainty_aware_hybrid_inference(prompt_text, uncertainty_threshold=0.5)
    print("Generated text:", generated_text)
    print("True skip ratio:", tsr)
    print("Transmission rate:", tsr)
