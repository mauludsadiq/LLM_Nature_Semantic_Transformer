#!/usr/bin/env python3
import os
import sys

def main():
    print("Downloading GPT-2 model from Hugging Face...")
    
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("Installing transformers...")
        os.system("pip3 install transformers torch")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    # Create cache directory
    os.makedirs(os.path.expanduser("~/.cache/huggingface/hub"), exist_ok=True)
    
    # Download model and tokenizer
    print("Downloading model (this may take a few minutes)...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Save in format tch can load
    import torch
    torch.save(model.state_dict(), 'gpt2_model.pt')
    tokenizer.save_pretrained('./')
    
    print("\n✓ Model downloaded successfully")
    print("  - gpt2_model.pt")
    print("  - tokenizer.json")
    print("  - merges.txt")
    print("  - vocab.json")
    print("\nModel size: ~500MB")

if __name__ == "__main__":
    main()
