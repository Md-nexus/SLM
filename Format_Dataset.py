from datasets import load_dataset

dataset = load_dataset("daily_dialog", trust_remote_code=True)

output_path = "data/brain.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for dialog in dataset["train"]:
        turns = dialog["dialog"]
        for i in range(len(turns) - 1):
            q = turns[i].strip().lower()
            a = turns[i + 1].strip().lower()
            if q and a:
                f.write(f"Q: {q}\nA: {a}\n===\n")

print(f"Dataset saved to {output_path}")
