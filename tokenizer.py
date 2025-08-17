import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from tokenizers.processors import TemplateProcessing

# Load the text data
with open("data/brain.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create a new tokenizer
tokenizer = Tokenizer(models.BPE())

# Configure the tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A: $B:1 </s>:1",
    special_tokens=[
        ("<s>", 0),
        ("</s>", 1),
    ],
)

# Train the tokenizer
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
    show_progress=True
)

# Train on the text data
tokenizer.train_from_iterator([text], trainer=trainer)

# Save the trained tokenizer
tokenizer.save("data/tokenizer.json")

# Test the tokenizer
encoded = tokenizer.encode(text[:1000])
print(f"Original text length: {len(text[:1000])}")
print(f"Encoded tokens: {len(encoded.tokens)}")
print(f"Sample tokens: {encoded.tokens[:20]}")

# Save vocabulary for compatibility
vocab = tokenizer.get_vocab()
stoi = vocab
itos = {v: k for k, v in vocab.items()}

# Save vocabulary in the same format as before for compatibility
with open("data/vocab.json", "w", encoding="utf-8") as f:
    json.dump({"stoi": stoi, "itos": itos}, f, indent=4, ensure_ascii=False)

# Encode the full text and save
encoded_text = tokenizer.encode(text)
with open("data/encoded.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(map(str, encoded_text.ids)))

print("Subword tokenization complete.")
print(f"Vocabulary size: {len(vocab)}")
print(f"Total tokens: {len(encoded_text.ids)}")
