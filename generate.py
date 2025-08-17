import torch
import json
from model.slm import SLM
from tokenizers import Tokenizer

# Load the tokenizer
tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model with new architecture
slm = SLM(
    vocab_size=vocab_size,
    embed_size=256,
    hidden_size=512,
    num_layers=3,
    dropout=0.1
).to(device)

# Load trained weights
slm.load_state_dict(torch.load("model/slm_weight.pt", map_location=device))
slm.eval()

def encode(text):
    """Encode text using the subword tokenizer"""
    encoded = tokenizer.encode(text)
    return encoded.ids

def decode(ids):
    """Decode token IDs back to text"""
    return tokenizer.decode(ids)

def generate_response(prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.9):
    """Generate response with improved sampling strategies"""
    # Encode the prompt
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate using the model's built-in generation method
    with torch.no_grad():
        generated_ids = slm.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            eos_token_id=tokenizer.token_to_id("</s>") if tokenizer.token_to_id("</s>") else None
        )
    
    # Decode the generated text
    full_text = decode(generated_ids[0].tolist())
    
    # Extract only the new part (after the prompt)
    prompt_tokens = len(input_ids[0])
    response = decode(generated_ids[0][prompt_tokens:].tolist())
    
    return response.strip()

def chat_loop():
    """Interactive chat loop with conversation memory"""
    conversation_history = []
    max_history = 3  # Keep last 3 exchanges for context
    
    print("🤖 Enhanced SLM Chat Bot")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to reset conversation history")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye! 👋")
            break
        elif user_input.lower() == "clear":
            conversation_history = []
            print("Conversation history cleared.")
            continue
        elif not user_input:
            continue
        
        # Build context from conversation history
        context = ""
        if conversation_history:
            context = " ".join(conversation_history[-max_history:]) + " "
        
        # Create the full prompt
        full_prompt = f"{context}Q: {user_input}\nA:"
        
        # Generate response
        try:
            response = generate_response(
                prompt=full_prompt,
                max_new_tokens=80,
                temperature=0.7,
                top_k=40,
                top_p=0.85
            )
            
            # Clean up the response
            response = response.replace("Q:", "").replace("A:", "").strip()
            if response.endswith("==="):
                response = response[:-3].strip()
            
            print(f"\nBot: {response}")
            
            # Update conversation history
            conversation_history.append(f"Q: {user_input}")
            conversation_history.append(f"A: {response}")
            
            # Keep only recent history
            if len(conversation_history) > max_history * 2:
                conversation_history = conversation_history[-max_history * 2:]
                
        except Exception as e:
            print(f"\nBot: Sorry, I encountered an error: {str(e)}")
        
        print("-" * 50)

if __name__ == "__main__":
    chat_loop()
