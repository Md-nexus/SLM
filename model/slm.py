import torch
import torch.nn as nn
import torch.nn.functional as F


class SLM(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Multi-layer LSTM for better sequence modeling
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism for better context understanding
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
        # Output projection with layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.shape
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq, embed_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(embedded, hidden)
        
        # Apply attention for better context understanding
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection and layer norm
        out = self.layer_norm(lstm_out + attn_out)
        out = self.dropout(out)
        
        # Project to vocabulary
        logits = self.fc(out)
        
        return logits, (h_n, c_n)
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.9, 
                 do_sample=True, pad_token_id=None, eos_token_id=None):
        """Enhanced generation with better sampling strategies"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize hidden state
        hidden = None
        
        for _ in range(max_new_tokens):
            # Forward pass
            logits, hidden = self.forward(input_ids, hidden)
            next_token_logits = logits[:, -1, :] / temperature
            
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return input_ids
