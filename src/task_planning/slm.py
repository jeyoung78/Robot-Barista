from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "./tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, ignore_mismatched_sizes=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# Receives input prompt, theta max, and K in token, float and int, returns draft token, uncertainty, and vocabulary distribution
def slm_inference(generated, theta_max: float = 2.0, K: int = 20):
    with torch.no_grad():
        outputs = model(generated)
    logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    draft_distribution = torch.softmax(next_token_logits, dim=-1)
    draft_token = torch.multinomial(draft_distribution, num_samples=1)
    draft_token_id = draft_token.item()
    draft_token_text = tokenizer.decode(draft_token_id).strip()

    perturbed_tokens = []
    perturbed_tokens_text = []
    temperatures = torch.FloatTensor(K).uniform_(0, theta_max)
    for temp in temperatures:
        perturbed_logits = next_token_logits / temp
        perturbed_distribution = torch.softmax(perturbed_logits, dim=-1)
        sampled_token = torch.multinomial(perturbed_distribution, num_samples=1)
        token_id = sampled_token.item()
        perturbed_tokens.append(token_id)
        perturbed_tokens_text.append(tokenizer.decode(token_id).strip())

    diff_count = sum(1 for token in perturbed_tokens if token != draft_token_id)
    uncertainty = diff_count / K

    return draft_token_id, uncertainty, draft_distribution


