#!/bin/bash
echo "Downloading GPT-2 model from Hugging Face..."
mkdir -p ~/.cache/huggingface/hub
python3 -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Save in format tch can load
torch.save(model.state_dict(), 'gpt2_model.pt')
tokenizer.save_pretrained('./')
print('Model downloaded successfully')
"
