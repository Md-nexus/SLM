#!/usr/bin/env python3
"""
Test script to verify the enhanced SLM setup
"""

import torch
import json
from tokenizers import Tokenizer
from model.slm import SLM

def test_tokenizer():
    """Test the subword tokenizer"""
    print("🧪 Testing tokenizer...")
    
    try:
        # Load tokenizer
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
        print(f"✅ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {vocab_size}")
        
        # Test encoding/decoding
        test_text = "Hello, how are you today?"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded.ids)
        
        print(f"   Test text: '{test_text}'")
        print(f"   Encoded tokens: {len(encoded.tokens)}")
        print(f"   Decoded: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_model():
    """Test the enhanced model architecture"""
    print("\n🧪 Testing model...")
    
    try:
        # Load tokenizer for vocab size
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
        
        # Create model
        model = SLM(
            vocab_size=vocab_size,
            embed_size=256,
            hidden_size=512,
            num_layers=3,
            dropout=0.1
        )
        
        print(f"✅ Model created successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        test_input = torch.randint(0, vocab_size, (2, 10)).to(device)
        output, hidden = model(test_input)
        
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Hidden state: {type(hidden)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_generation():
    """Test the generation capabilities"""
    print("\n🧪 Testing generation...")
    
    try:
        # Load tokenizer
        tokenizer = Tokenizer.from_file("data/tokenizer.json")
        vocab_size = tokenizer.get_vocab_size()
        
        # Create model
        model = SLM(
            vocab_size=vocab_size,
            embed_size=256,
            hidden_size=512,
            num_layers=3,
            dropout=0.1
        )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Test generation
        test_prompt = "Hello, how are you?"
        input_ids = torch.tensor(tokenizer.encode(test_prompt).ids, dtype=torch.long).unsqueeze(0).to(device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"✅ Generation test successful")
        print(f"   Prompt: '{test_prompt}'")
        print(f"   Generated: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Enhanced SLM Tests")
    print("=" * 50)
    
    tests = [
        test_tokenizer,
        test_model,
        test_generation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    if all(results):
        print("✅ All tests passed! Your enhanced SLM is ready to go!")
        print("\n🎯 Next steps:")
        print("1. Run: python GET_Data.py (to get better dataset)")
        print("2. Run: python tokenizer.py (to create subword tokenizer)")
        print("3. Run: python train.py (to train the enhanced model)")
        print("4. Run: python generate.py (to chat with your bot)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        failed_tests = [i+1 for i, result in enumerate(results) if not result]
        print(f"   Failed tests: {failed_tests}")

if __name__ == "__main__":
    main()
