from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
ADAPTER_PATH    = "./models/tiny-llama-mega"

_tokenizer = None
_model     = None

def _init_slm():
    global _tokenizer, _model
    if _model is not None:
        return

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=False
    )
    _model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        low_cpu_mem_usage=True,     # optional
        torch_dtype=torch.float16   # optional; speeds things up
    )
    _model = PeftModel.from_pretrained(_model, ADAPTER_PATH)

    _model.to(DEVICE)
    _model.eval()

def slm_inference(generated, allowed_tokens, theta_max: float = 2.0, K: int = 20):
    _init_slm()

    with torch.no_grad():
        outputs = _model(generated)

    logits = outputs.logits
    next_token_logits = logits[0, -1, :]

    allowed_mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
    allowed_token_ids = torch.tensor(allowed_tokens, device=next_token_logits.device)
    allowed_mask[allowed_token_ids] = True

    disallowed_value = 0
    masked_logits = torch.where(allowed_mask, next_token_logits, torch.tensor(disallowed_value, device=next_token_logits.device))

    draft_distribution = torch.softmax(masked_logits, dim=-1)
    draft_token = torch.multinomial(draft_distribution, num_samples=1)
    draft_token_id = draft_token.item()
    # draft_token_text = tokenizer.decode(draft_token_id).strip()

    perturbed_tokens = []
    perturbed_tokens_text = []
    temperatures = torch.FloatTensor(K).uniform_(0, theta_max)
    for temp in temperatures:
        perturbed_logits = next_token_logits / temp
        perturbed_distribution = torch.softmax(perturbed_logits, dim=-1)
        sampled_token = torch.multinomial(perturbed_distribution, num_samples=1)
        token_id = sampled_token.item()
        perturbed_tokens.append(token_id)
        perturbed_tokens_text.append(_tokenizer.decode(token_id).strip())

    diff_count = sum(1 for token in perturbed_tokens if token != draft_token_id)
    uncertainty = diff_count / K

    return draft_token_id, uncertainty, draft_distribution

