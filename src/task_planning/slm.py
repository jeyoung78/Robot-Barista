from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import serial
import json
import time

SERIAL_PORT = 'COM3'  # Use the appropriate USB port identifier
BAUDRATE = 9600

device = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map=device,        
)

ADAPTER_PATH = "./tinyllama-finetuned"  # your LoRA/PEFT checkpoint
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

model_dir = "./tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

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

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except Exception as e:
        print("Error opening serial port:", e)
        return

    print("Listener active and waiting for commands...")
    
    while True:
        try:
            if ser.in_waiting > 0:
                command_line = ser.readline().decode().strip()
                print("Received command:", command_line)

                try:
                    command = json.loads(command_line)
                    if command.get("function") == "slm_inference":
                        params = command.get("params", [])
                        if len(params) >= 1:
                            generated_list = params[0]
                            theta_max = params[1] if len(params) > 1 else 2.0
                            K = params[2] if len(params) > 2 else 20
                            
                            generated_tensor = torch.tensor(generated_list, device=device).unsqueeze(0)
                            
                            draft_token_id, uncertainty, draft_distribution = slm_inference(generated_tensor, theta_max, K)
                            
                            response = {
                                "draft_token_id": int(draft_token_id),
                                "uncertainty": uncertainty,
                                "draft_distribution": draft_distribution.tolist() if hasattr(draft_distribution, "tolist") else draft_distribution
                            }
                        else:
                            response = {"error": "Insufficient parameters."}
                    else:
                        response = {"error": "Unknown function."}
                except Exception as ex:
                    response = {"error": str(ex)}
                
                response_str = json.dumps(response) + "\n"
                ser.write(response_str.encode())
                print("Sent response:", response_str.strip())
        except Exception as e:
            print("Communication error:", e)

if __name__ == "__main__":
    main()