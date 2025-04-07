import torch

from slm import slm_inference
from llm import llm_verification
from transformers import AutoTokenizer

model_dir = "./tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

def detokenize(token_id):
    if type(token_id) == int:
        return tokenizer.decode(token_id)
    else:
        return tokenizer.decode(token_id[0], skip_special_tokens=True)

# returns final prompt and true skip ratio
def uncertainty_aware_hybrid_inference(prompt: str, max_new_tokens: int = 100, uncertainty_threshold: float = 0.5):
    resample = 0
    num_inference = 0
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token} "
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    for i in range(0, max_new_tokens):
        num_inference = i
        draft_token_id, uncertainty, draft_distribution = slm_inference(generated=generated)
        if uncertainty > uncertainty_threshold:
            final_token_id, accepted = llm_verification(draft_distribution, draft_token_id, generated)
            chosen_token = torch.tensor([final_token_id], device=device)
            if not accepted:
                resample = resample + 1
        else: 
            final_token_id = draft_token_id
        chosen_token = torch.tensor([final_token_id], device=device)
        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)
        # print(detokenize(generated))
        if detokenize(chosen_token) == "Done":
            break

    return detokenize(generated)[initial_length:], (1 - resample/num_inference)

if __name__ == "__main__":
    prompt_text = "Can I have vanilla latte?"
    generated, tsr = uncertainty_aware_hybrid_inference(prompt_text)
    print(generated, tsr)
