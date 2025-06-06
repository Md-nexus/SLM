import torch
import json
from model.slm import SLM

with open("data/vocab.json", "r", encoding="utf-8") as f:
	vocab = json.load(f)

stoi = vocab["stoi"]
itos = {int(i): ch for i, ch in vocab["itos"].items()}
vocab_size = len(stoi)

device = "cuda" if torch.cuda.is_available() else "cpu"

slm = SLM(vocab_size, embed_size=128, hidden_size=256).to(device)
slm.load_state_dict(torch.load("model/slm_weight.pt", map_location=device))
slm.eval()

def local_encode(text):
	return [stoi.get(c, 0) for c in text]

def local_decode(indices):
	return ''.join([itos.get(i, '') for i in indices])


@torch.no_grad()
def generate(seed_text, max_new_tokens=100, temperature=0.8, stop_token="==="):
    input_ids = torch.tensor(local_encode(seed_text), dtype=torch.long).unsqueeze(0).to(device)
    hidden = None

    for _ in range(max_new_tokens):
        output, hidden = slm(input_ids, hidden)
        logits = output[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        decoded = local_decode(input_ids[0].tolist())
        if stop_token in decoded:
            break

    final = local_decode(input_ids[0].tolist())
    return final.split(stop_token)[0].strip()

if __name__ == "__main__":
    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ["quit", "exit"]:
            break
        full_prompt = f"\nQ: {prompt}\nA:"
        response = generate(full_prompt, max_new_tokens=100, temperature=0.7)
        print("\nBot:", response.split('A:')[-1].strip())
        print("-" * 50)
