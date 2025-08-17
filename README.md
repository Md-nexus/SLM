# Enhanced SLM (Simple Language Model) 🤖

A significantly improved conversational AI model designed to reduce hallucination and provide more coherent responses.

## 🚀 Key Improvements Made

### 1. **Subword Tokenization** 
- **Before**: Character-level tokenization (every letter = 1 token)
- **After**: Byte-Pair Encoding (BPE) with 8000 vocabulary size
- **Impact**: Eliminates broken words, better language understanding

### 2. **Enhanced Model Architecture**
- **Before**: Simple GRU with 128/256 hidden size
- **After**: Multi-layer LSTM (3 layers) with 256/512 hidden size + Attention
- **Features**: 
  - Multi-head attention mechanism
  - Layer normalization
  - Dropout for regularization
  - Proper weight initialization

### 3. **Better Training Strategy**
- **Before**: Basic training with fixed learning rate
- **After**: 
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate scheduler
  - Gradient clipping
  - Early stopping with validation
  - Train/validation split (90/10)

### 4. **Improved Generation**
- **Before**: Basic greedy sampling
- **After**: 
  - Top-k filtering (k=50)
  - Top-p (nucleus) sampling (p=0.9)
  - Temperature control
  - Conversation memory (last 3 exchanges)

### 5. **Enhanced Dataset**
- **Before**: DailyDialog only (~13k conversations)
- **After**: DailyDialog + OpenAssistant (much larger, more diverse)
- **Features**: Better filtering, more varied conversation styles

## 📁 Project Structure

```
SLM/
├── data/
│   ├── brain.txt          # Training dataset
│   ├── vocab.json         # Vocabulary mapping
│   ├── encoded.txt        # Tokenized data
│   └── tokenizer.json     # Subword tokenizer
├── model/
│   ├── slm.py            # Enhanced model architecture
│   └── slm_weight.pt     # Trained weights
├── GET_Data.py           # Dataset collection
├── tokenizer.py          # Subword tokenizer creation
├── train.py              # Enhanced training script
├── generate.py           # Improved generation script
├── test_setup.py         # Setup verification
└── requirements.txt      # Dependencies
```

## 🛠️ Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Get enhanced dataset**:
   ```bash
   python GET_Data.py
   ```

3. **Create subword tokenizer**:
   ```bash
   python tokenizer.py
   ```

4. **Train the model**:
   ```bash
   python train.py
   ```

5. **Chat with your bot**:
   ```bash
   python generate.py
   ```

## 🧪 Testing

Run the test suite to verify everything is working:
```bash
python test_setup.py
```

## 🎯 Key Features

### Reduced Hallucination
- **Subword tokenization** prevents broken word generation
- **Larger context window** (128 tokens vs 64) for better memory
- **Attention mechanism** helps focus on relevant context
- **Better sampling strategies** reduce repetitive/nonsensical outputs

### Improved Coherence
- **Conversation memory** maintains context across exchanges
- **Multi-layer architecture** captures more complex patterns
- **Validation-based training** prevents overfitting
- **Enhanced dataset** provides better training examples

### Better User Experience
- **Interactive chat loop** with conversation history
- **Error handling** for graceful failures
- **Clear commands** (quit, clear, etc.)
- **Progress indicators** during training

## 🔧 Configuration

### Model Parameters
```python
# Enhanced model configuration
embed_size = 256      # Increased from 128
hidden_size = 512     # Increased from 256
num_layers = 3        # Multi-layer architecture
dropout = 0.1         # Regularization
```

### Training Parameters
```python
seq_length = 128      # Increased context window
batch_size = 16       # Optimized for larger model
learning_rate = 0.0001
weight_decay = 0.01   # L2 regularization
```

### Generation Parameters
```python
temperature = 0.7     # Controls randomness
top_k = 40           # Top-k filtering
top_p = 0.85         # Nucleus sampling
max_new_tokens = 80  # Response length
```

## 📊 Performance Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Vocab Size | ~75 chars | 8000 subwords | 106x larger |
| Context Window | 64 tokens | 128 tokens | 2x larger |
| Model Parameters | ~500K | ~2.5M | 5x larger |
| Architecture | GRU | LSTM + Attention | More sophisticated |
| Sampling | Greedy | Top-k + Top-p | Better diversity |

## 🚨 Troubleshooting

### Common Issues

1. **"No module named 'torch'"**
   - Run: `pip install torch numpy tqdm datasets transformers`

2. **Tokenizer not found**
   - Run: `python tokenizer.py` first

3. **Out of memory during training**
   - Reduce `batch_size` in `train.py`
   - Use CPU instead of GPU

4. **Poor generation quality**
   - Train for more epochs
   - Adjust temperature/top-k/top-p parameters
   - Check dataset quality

## 🎓 Learning Resources

- [Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Attention Mechanisms](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
- [Language Model Training](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is open source and available under the MIT License.
