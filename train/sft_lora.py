import os
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_from_disk, load_dataset
import transformers
import trl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
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
    def __init__(self, hidden_size, num_loras):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_loras),
        )
        self.cached_probs = None

    def forward(self, x):
        pooled = x[:, 0]
        logits = self.router(pooled.to(self.router[0].weight.dtype))
        self.cached_probs = torch.softmax(logits, dim=-1)[0]  # shape: [num_loras]
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
        lora_outputs = []
        for i, lora in enumerate(self.lora_list):
            lora_out = lora(x) * probs[i]  # Scale output by router probability
            lora_outputs.append(lora_out)
            
        # Sum all lora outputs
        lora_contribution = sum(lora_outputs)
        
        # Add to base output
        return out + lora_contribution

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

    def __init__(self, model, router_cutoff_layer=2, num_loras=4, lora_r=8, lora_alpha=16):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_layers = len(model.model.layers)
        self.router_cutoff = router_cutoff_layer

        self.freeze_base_model()  # Freeze base model

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_trainable:,}")

        self.router = GlobalRouter(model.config.hidden_size, num_loras)
        self.modified_layers = nn.ModuleList()

        for i, block in enumerate(model.model.layers):
            if i >= self.router_cutoff:
                mlp = block.mlp
                lora_list = [LoRALayer(model.config.hidden_size, model.config.hidden_size, lora_r, lora_alpha) for _ in range(num_loras)]
                new_ffn = LayerWithLoRAMixture(mlp, lora_list, router_ref=self.router)
                block.mlp = new_ffn
                self.modified_layers.append(new_ffn)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.model.model.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        extended_attention_mask = expand_attention_mask(attention_mask, dtype=hidden.dtype, tgt_len=seq_len)

        for i in range(self.router_cutoff):
            layer_outputs = self.model.model.layers[i](
                hidden,
                attention_mask=extended_attention_mask,
                position_ids=position_ids
            )
            hidden = layer_outputs[0]

        _ = self.router(hidden)  # sets cached_probs

        for i in range(self.router_cutoff, self.num_layers):
            layer_outputs = self.model.model.layers[i](
                hidden,
                attention_mask=extended_attention_mask,
                position_ids=position_ids
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
            return {'loss': loss, 'logits': hidden}

        return {'logits': hidden}

@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    block_size: int = field(default=32768)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='simplescaling/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name)
    wrapped_model = QwenRouterWrapper(model, router_cutoff_layer=2)
    
    # Count trainable parameters
    trainable_params = 0
    all_params = 0
    for name, param in wrapped_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            print('Trainable:', name)
            trainable_params += param.numel()
    logging.info(f"Trainable parameters: {trainable_params:,d} ({trainable_params/all_params:.2%} of all parameters)")

    # Load dataset
    dataset = load_dataset(config.train_file_path)

    # Setting up the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        tokenizer.pad_token = "<|fim_pad|>"

    # Data collator for completion-only LM
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    
    args.dataset_text_field = 'text'
    args.max_seq_length = config.block_size
    
    # Initialize trainer with our MoLoRA model
    trainer = trl.SFTTrainer(
        wrapped_model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
        # No peft_config since we're handling adapter setup ourselves
    )

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()