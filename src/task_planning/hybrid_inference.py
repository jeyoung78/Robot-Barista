import torch
import time
import requests  
from slm import slm_inference
from transformers import AutoTokenizer

model_dir = "./tinyllama-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# Matcha Frappuccino: 1. Place Cup 2. Pour Milk 3. Add Matcha Powder 4. Add Sugar 5. Add Ice 6. Blend Beverage 7. Add Whipped Cream 8. Serve Beverage 9. Done
# Cinnamon Dolce Latte: 1. Place Cup 2. Pour Espresso 3. Pour Milk 4. Pour Cinnamon Dolce Syrup 5. Add Whipped Cream 6. Serve Beverage 7. Done
# Pumking Spice Latte: 1. Place Cup 2. Pour Espresso 3. Pour Pumpkin Spice Syrup 4. Pour Milk 5. Add Whipped Cream 6. Serve Beverage 7. Done
# Caramel Frozen Blended Coffee: 1. Place Cup 2. Pour Espresso 4. Pour Milk 5. Pour Salt 6. Pour Whipped Cream 7. Serve Beverage 8. Done
# Vanilla Latte: 1. Place Cup 2. Pour Milk 3. Pour Vanilla Syrup 4. Serve Beverage 5. Done
# Mint Chocolate Mocha: 1. Place Cup 2. Pour Chocolate 3. Pour Mint 4. Pour Whipped Cream 5. Serve Beverage 6. Done
# Caramel Honey Latte: 1. 
# allowed_text = "1. Place Cup 2. Pour Chocolate 3. Pour Mint 4. Pour Whipped Cream 5. Serve Beverage Done"
# temp = tokenizer.encode(allowed_text, add_special_tokens=False)

allowed_tokens = [260, 261, 263, 265, 267, 269, 271, 273, 274, 275, 279, 280, 282, 286, 287, 288, 290, 291, 293, 295, 298, 301, 304, 306, 309, 314, 315, 317, 319, 322, 323, 327, 329, 330, 331, 335, 341, 342, 345, 349, 350, 360, 364, 
                  365, 371, 374, 379, 382, 383, 385, 387, 388, 390, 391, 394, 398, 402, 403, 405, 411, 412, 432, 435, 447, 454, 457, 465, 468, 482, 496, 498, 503, 505, 508, 521, 524, 535, 542, 558, 562, 601, 603, 617, 625, 653, 661, 672, 
                  678, 679, 687, 690, 694, 696, 715, 719, 728, 747, 763, 796, 799, 805, 831, 837, 838, 879, 907, 912, 932, 936, 949, 1025, 1029, 1050, 1085, 1109, 1173, 1219, 1236, 1296, 1302, 1310, 1358, 1383, 1410, 1451, 1505, 1507, 1559, 1581, 1633, 
                  1648, 1701, 1704, 1706, 1725, 1766, 1789, 1794, 1797, 1816, 1847, 1858, 1886, 1920, 1943, 1993, 2049, 2078, 2088, 2139, 2148, 2163, 2276, 2292, 2362, 2379, 2496, 2537, 2559, 2610, 2635, 2638, 2646, 2730, 2753, 2795, 2878, 2910, 2911, 
                  3113, 3118, 3189, 3252, 3423, 3427, 3462, 3478, 3623, 3712, 3833, 3848, 3880, 3938, 3958, 3973, 4094, 4161, 4326, 4367, 4524, 4584, 4628, 4692, 4712, 4764, 4798, 4805, 4886, 4981, 4989, 4992, 5104, 5260, 5383, 5617, 5621, 5678, 5815, 
                  5881, 5962, 5990, 6054, 6062, 6235, 6241, 6324, 6527, 6536, 6556, 6803, 6978, 7013, 7053, 7254, 7315, 7347, 7537, 7612, 7646, 7655, 7740, 7933, 8037, 8168, 8280, 8296, 8315, 8626, 8668, 8836, 8927, 8971, 9089, 9160, 9216, 9243, 9817, 
                  9878, 9892, 9897, 9969, 10293, 10484, 10492, 10750, 10765, 10924, 11492, 11548, 11625, 12019, 12113, 12846, 13163, 13209, 13231, 13737, 14008, 14269, 14351, 14514, 14596, 14890, 14954, 15043, 15065, 15484, 15816, 15847, 15935, 16108, 
                  16267, 16344, 16699, 17042, 17184, 17354, 17827, 18254, 19605, 19710, 20290, 20559, 20814, 21144, 21193, 21261, 21265, 21305, 21353, 21601, 21783, 21881, 21954, 22181, 22531, 23167, 23212, 23429, 23816, 24841, 25589, 25606, 25679, 26048, 
                  26163, 26438, 26515, 26731, 27274, 29871, 29872, 29874, 29877, 29881, 29884, 29885, 29888, 29889, 29890, 29892, 29893, 29895, 29896, 29899, 29900, 29906, 29915, 29920, 29924, 29926, 29929, 29941, 29945, 29946, 29947, 29953, 29955, 29973]

'''
for element in temp:
    if element not in allowed_tokens:
        allowed_tokens.append(element)
        print(element)
'''

def detokenize(token_id):
    if isinstance(token_id, int):
        return tokenizer.decode(token_id)
    else:
        return tokenizer.decode(token_id[0], skip_special_tokens=True)


# allowed_tokens = list(set(allowed_tokens))
# print(len(allowed_tokens))
# print(allowed_tokens)

# for element in allowed_tokens:
#     print(detokenize(element))

# Returns the final generated prompt and the true skip ratio.
def uncertainty_aware_hybrid_inference(prompt: str, max_new_tokens: int = 100, uncertainty_threshold: float = 0.5, verbose: bool = True):
    start = time.time()
    resample = 0
    num_inference = 0
    num_transmission = 0
    special_token = "<recipe_generation>"
    prompt = f"{special_token} {prompt} {special_token}"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = inputs["input_ids"]

    initial_length = len(prompt)

    for i in range(max_new_tokens):
        num_inference = i + 1
        draft_token_id, uncertainty, draft_distribution = slm_inference(generated=generated, allowed_tokens=allowed_tokens)
        
        if uncertainty > uncertainty_threshold:
            num_transmission = num_transmission + 1
            if verbose:
                print(f"High uncertainty ({uncertainty:.2f} > {uncertainty_threshold}); calling remote LLM verification...")
            # Convert draft_distribution to a list if needed.
            if isinstance(draft_distribution, torch.Tensor):
                draft_distribution_list = draft_distribution.tolist()
            else:
                draft_distribution_list = draft_distribution
            
            payload = {
                "draft_distribution": draft_distribution_list,
                "draft_token_id": int(draft_token_id),
                "generated": generated.squeeze(0).tolist(),
                "allowed_tokens": allowed_tokens
            }
            
            server_url = "http://165.132.40.52:5001/llm_verification"
            try:
                response = requests.post(server_url, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()
                final_token_id = data['result_token_id']
                accepted = data['accepted']
                if not accepted:
                    resample += 1
            except Exception as e:
                if verbose:
                    print("Error calling remote llm_verification:", e)
                final_token_id = draft_token_id
        else:
            if verbose:
                print(f"Low uncertainty ({uncertainty:.2f} <= {uncertainty_threshold}); using SLM token directly.")
            final_token_id = draft_token_id

        chosen_token = torch.tensor([final_token_id], device=device)
        generated = torch.cat([generated, chosen_token.unsqueeze(0)], dim=1)
        if verbose:
            print(detokenize(generated))

        if detokenize(chosen_token).strip() == "Done":
            break

    tsr = (1 - resample/num_inference) if num_inference > 0 else 1.0
    tr = num_transmission/num_inference if num_inference > 0 else 1.0
    time_elapsed = time.time() - start
    return detokenize(generated)[initial_length:], tsr, tr, num_inference, time_elapsed

if __name__ == "__main__":
    prompt_text = "May I order a Caramel Ginger Espresso??"
    generated_text, tsr, tr, num_inference, time_elapsed = uncertainty_aware_hybrid_inference(prompt_text, uncertainty_threshold=2)
    print("Generated text:", generated_text)
    print("True skip ratio:", tsr)
    print("Transmission rate:", tsr)
