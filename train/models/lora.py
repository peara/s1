import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.modeling_utils import PreTrainedModel

# === LoRA and Router Definitions ===

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        # === Proper initialization ===
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling


class GlobalRouter(nn.Module):
    def __init__(self, hidden_size, num_loras, temperature=1.0, diversity_weight=0.1, expert_dropout=0.2, top_k=2):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_loras),
        )
        self.cached_probs = None
        self.temperature = temperature
        self.diversity_weight = diversity_weight
        self.num_loras = num_loras
        self.expert_dropout = expert_dropout  # Probability of dropping each expert
        self.top_k = min(top_k, num_loras)  # Ensure top_k is not greater than num_loras
        
        # Register buffers to track usage statistics (will be saved with model)
        self.register_buffer('usage_count', torch.zeros(num_loras))
        self.register_buffer('total_samples', torch.tensor(0.))
        self.last_diversity_loss = 0.0

    def forward(self, x, return_loss=False):
        pooled = x[:, 0]
        logits = self.router(pooled.to(self.router[0].weight.dtype))
        
        # Apply expert dropout during training
        if self.training and self.expert_dropout > 0:
            # Create a dropout mask - randomly set some logits to -inf
            # Importantly, we ensure at least one expert remains active
            batch_size = pooled.shape[0]
            dropout_mask = torch.ones_like(logits, device=logits.device)
            
            # For each example in the batch
            for i in range(batch_size):
                # Randomly select which experts to drop
                drop_mask = torch.rand(self.num_loras, device=logits.device) < self.expert_dropout
                
                # Ensure we don't drop all experts (always keep at least one)
                if drop_mask.all():
                    # Keep one random expert
                    keep_idx = torch.randint(0, self.num_loras, (1,), device=logits.device)
                    drop_mask[keep_idx] = False
                
                # Ensure we keep at least top_k experts (don't drop too many)
                if drop_mask.sum() > self.num_loras - self.top_k:
                    # Find top-k experts
                    _, top_indices = torch.topk(logits[i], self.top_k)
                    # Make sure they're not dropped
                    drop_mask[top_indices] = False
                
                # Set the logits of dropped experts to -inf
                dropout_mask[i, drop_mask] = float('-inf')
            
            # Apply the mask
            logits = logits + dropout_mask
        
        # Apply temperature scaling
        scaled_logits = logits / self.temperature
        
        # Implement top-k routing - only keep top k logits per example
        batch_size = pooled.shape[0]
        top_k_mask = torch.zeros_like(scaled_logits, device=scaled_logits.device) - float('inf')
        
        for i in range(batch_size):
            # Find the indices of the top-k values
            top_k_values, top_k_indices = torch.topk(scaled_logits[i], self.top_k)
            # Place the top-k values in the mask
            top_k_mask[i, top_k_indices] = top_k_values
        
        # Apply softmax only over the top-k logits
        probs = torch.softmax(top_k_mask, dim=-1)
        self.cached_probs = probs[0]  # shape: [num_loras]
        
        # Update usage statistics when training
        if self.training:
            with torch.no_grad():
                # Add current probabilities to running count (for all examples in batch)
                batch_usage = probs.mean(dim=0)
                self.usage_count += batch_usage.detach()
                self.total_samples += 1
        
        # Calculate diversity loss if requested
        if return_loss and self.training and self.total_samples > 0:
            # Get the historical usage distribution
            usage_dist = self.usage_count / self.total_samples
            
            # Calculate KL divergence from uniform distribution
            # Higher KL = less uniform = more diversity loss
            uniform_dist = torch.ones_like(usage_dist) / self.num_loras
            kl_div = torch.sum(usage_dist * torch.log((usage_dist + 1e-10) / (uniform_dist + 1e-10)))
            
            # Calculate load balancing loss (penalize deviation from uniform)
            # This is based on the Switch Transformer approach
            router_prob_per_expert = probs.mean(dim=0)  # Mean across batch
            router_prob_per_expert = router_prob_per_expert.clamp(min=1e-10)  # Avoid zeros
            load_balancing_loss = self.num_loras * torch.sum(router_prob_per_expert * router_prob_per_expert)
            
            # Combine KL divergence with load balancing loss
            diversity_loss = self.diversity_weight * (kl_div + 0.5 * load_balancing_loss)
            self.last_diversity_loss = diversity_loss.item()
            
            return self.cached_probs, diversity_loss
            
        return self.cached_probs


class LayerWithLoRAMixture(nn.Module):
    def __init__(self, base_layer, lora_list, router_ref):
        super().__init__()
        self.base_layer = base_layer
        self.lora_list = nn.ModuleList(lora_list)
        self._router_ref = router_ref

    def forward(self, x):
        out = self.base_layer(x)
        probs = self._router_ref.cached_probs  # shape: [num_loras]
        if probs is None:
            return out
            
        # Process each LoRA separately and combine
        # Only active LoRAs (those with non-zero probability) will contribute
        lora_outputs = []
        for i, lora in enumerate(self.lora_list):
            if probs[i] > 0:  # Only process LoRAs with non-zero probability
                lora_out = lora(x) * probs[i]  # Scale output by router probability
                lora_outputs.append(lora_out)
            
        # Sum all lora outputs
        if lora_outputs:  # Check if there are any outputs to sum
            lora_contribution = sum(lora_outputs)
            # Add to base output
            return out + lora_contribution
        else:
            return out

# === Utility ===

def expand_attention_mask(attention_mask, dtype, tgt_len=None):
    batch_size, seq_len = attention_mask.shape
    if tgt_len is None:
        tgt_len = seq_len

    causal_mask = torch.tril(torch.ones((tgt_len, seq_len), dtype=dtype, device=attention_mask.device))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    expanded_mask = attention_mask[:, None, None, :] * causal_mask
    expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    return expanded_mask


# === Router-Integrated Model Wrapper ===

class QwenRouterWrapper(nn.Module):
    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        print("Base model frozen.")

    def __init__(self, model, router_cutoff_layer=2, num_loras=4, lora_r=8, lora_alpha=16, temperature=1.0, 
                 diversity_weight=0.1, expert_dropout=0.2, top_k=2):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_layers = len(model.model.layers)
        self.router_cutoff = router_cutoff_layer

        self.freeze_base_model()  # Freeze base model

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_trainable:,}")

        self.router = GlobalRouter(model.config.hidden_size, num_loras, temperature, diversity_weight, 
                                  expert_dropout, top_k)
        self.modified_layers = nn.ModuleList()

        for i, block in enumerate(model.model.layers):
            if i >= self.router_cutoff:
                for block_name in ['q_proj', 'k_proj', 'v_proj']:
                    mlp = getattr(block.self_attn, block_name)
                    lora_list = [LoRALayer(mlp.in_features, mlp.out_features, lora_r, lora_alpha) for _ in range(num_loras)]
                    new_ffn = LayerWithLoRAMixture(mlp, lora_list, router_ref=self.router)
                    setattr(block.self_attn, block_name, new_ffn)
                    self.modified_layers.append(new_ffn)

    """
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pre-compute router probabilities
        with torch.no_grad():
            pre_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        router_hidden = pre_outputs.hidden_states[self.router_cutoff]
        _ = self.router(router_hidden)
        
        # Now do the actual forward pass with router probabilities set
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs
    """

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.model.model.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = expand_attention_mask(attention_mask, dtype=hidden.dtype, tgt_len=seq_len)
        
        # Get rotary position embeddings
        # For Qwen2 models, this creates the cos/sin tensors needed by attention layers
        position_embeddings = self.model.model.rotary_emb(x=hidden, position_ids=position_ids)

        for i in range(self.router_cutoff):
            layer_outputs = self.model.model.layers[i](
                hidden,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings  # Pass the rotary embeddings
            )
            hidden = layer_outputs[0]

        # Get router probabilities and diversity loss when training
        router_output = self.router(hidden, return_loss=self.training)
        if self.training and isinstance(router_output, tuple):
            # When training, we get both probabilities and diversity loss
            _, diversity_loss = router_output
        else:
            # During inference, we only get probabilities
            diversity_loss = 0.0

        for i in range(self.router_cutoff, self.num_layers):
            layer_outputs = self.model.model.layers[i](
                hidden,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings  # Pass the rotary embeddings
            )
            hidden = layer_outputs[0]
        
        if hasattr(self.model.model, 'norm'):
            hidden = self.model.model.norm(hidden)

        hidden = self.model.lm_head(hidden)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            shift_logits = hidden[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add diversity loss to the main loss
            total_loss = loss + diversity_loss
            return {'loss': total_loss, 'logits': hidden}

        return {'logits': hidden}
