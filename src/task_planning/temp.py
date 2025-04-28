import requests

# ── CONFIG ────────────────────────────────────────────────────────────────
SERVER_HOST = "165.132.40.52"     # your Flask host
SERVER_PORT = 5002
ENDPOINT    = f"http://{SERVER_HOST}:{SERVER_PORT}/hybrid_inference_partial"

# ── PAYLOAD SETUP ────────────────────────────────────────────────────────
payload = {
    "prompt": "<recipe_generation> Make me a latte with oat milk <recipe_generation>",
    "max_new_tokens": 50,
    "uncertainty_threshold": 0.4,
    "verbose": True
}

count = 2

prompt = "<recipe_generation> Make me a latte with oat milk <recipe_generation> 1."

while count < 10:
    payload = {
        "prompt": prompt,
        "max_new_tokens": 50,
        "uncertainty_threshold": 0.4,
        "verbose": True
    }

    try:
        resp = requests.post(ENDPOINT, json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        exit(1)

    # ── RESPONSE HANDLING ────────────────────────────────────────────────────
    # print("New Step Generated:\n", result["partial_text"])
    # print(f"True-Skip Ratio:      {result['true_skip_ratio']:.2f}")
    # print(f"Transmission Rate:    {result['transmission_rate']:.2f}")
    # print(f"Tokens Inferred:      {result['num_inference']}")
    # print(f"Latency (seconds):    {result['latency_s']:.3f}")

    prompt = prompt + result["partial_text"] + str(count)
    print(prompt)

    if "done" in prompt.lower():
        break

    count = count + 1