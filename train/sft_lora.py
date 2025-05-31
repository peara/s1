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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import wandb

# === Router Analysis Callback ===

class RouterAnalysisCallback(transformers.TrainerCallback):
    """Callback to monitor router selection during training."""
    
    def __init__(self, model, tokenizer, dataset, num_examples=10, log_freq=100, log_dir='router_logs'):
        self.model = model
        self.tokenizer = tokenizer
        self.log_freq = log_freq
        self.log_dir = log_dir
        self.num_examples = num_examples
        os.makedirs(log_dir, exist_ok=True)
        
        # Select a few examples from the dataset for monitoring
        indices = np.random.choice(len(dataset), num_examples, replace=False)
        self.example_texts = []
        self.example_inputs = []
        
        for idx in indices:
            example = dataset[idx]
            text = example['text'] if 'text' in example else str(example)
            self.example_texts.append(text[:100] + "..." if len(text) > 100 else text)
            
            # Tokenize the example
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            self.example_inputs.append(inputs)
        
        # Track distributions over time
        self.router_logs = []
        
        # Create output files
        self.csv_file = os.path.join(log_dir, 'router_activations.csv')
        with open(self.csv_file, 'w') as f:
            num_loras = getattr(model.router.router[-1], 'out_features', 1)
            header = ['step', 'example_id', 'text']
            for i in range(num_loras):
                header.append(f'lora_{i}')
            header.append('temperature')
            header.append('entropy')
            f.write(','.join(header) + '\n')
    
    def get_router_probs(self, input_dict):
        """Get router probabilities for a given input."""
        input_ids = input_dict['input_ids'].to(self.model.device)
        
        # Extract hidden states for router
        with torch.no_grad():
            # Process through model until router cutoff
            hidden = self.model.model.model.embed_tokens(input_ids)
            
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            extended_mask = expand_attention_mask(attention_mask, dtype=hidden.dtype)
            position_embeddings = self.model.model.model.rotary_emb(x=hidden, position_ids=position_ids)
            
            # Run through initial layers
            for i in range(self.model.router_cutoff):
                layer_outputs = self.model.model.model.layers[i](
                    hidden,
                    attention_mask=extended_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )
                hidden = layer_outputs[0]
            
            # Get router probabilities
            probs = self.model.router(hidden).detach().cpu().numpy()
            
        return probs
    
    def on_log(self, args, state, control, **kwargs):
        """Log router activations periodically."""
        # Only log at specified frequency
        if state.global_step % self.log_freq != 0:
            return
            
        step = state.global_step
        logging.info(f"Logging router activations at step {step}")
        
        step_logs = []
        
        # Get router probabilities for each example
        for ex_id, inputs in enumerate(self.example_inputs):
            try:
                probs = self.get_router_probs(inputs)
                temperature = getattr(self.model.router, 'temperature', 1.0)
                
                # Calculate entropy of distribution
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                
                # Log to CSV
                with open(self.csv_file, 'a') as f:
                    row = [str(step), str(ex_id), f'"{self.example_texts[ex_id]}"'] 
                    row.extend([str(p) for p in probs.flatten()])
                    row.append(str(temperature))
                    row.append(str(entropy))
                    f.write(','.join(row) + '\n')
                
                # Save for wandb logging
                step_logs.append({
                    'step': step,
                    'example_id': ex_id,
                    'text': self.example_texts[ex_id],
                    'probs': probs.flatten().tolist(),
                    'temperature': temperature,
                    'entropy': entropy
                })
                    
            except Exception as e:
                logging.warning(f"Error logging router activations for example {ex_id}: {e}")
        
        self.router_logs.extend(step_logs)
        
        # Log to wandb if it's active
        if wandb.run is not None:
            # Log raw data
            wandb.log({
                "router/temperature": temperature,
                "router/avg_entropy": np.mean([log['entropy'] for log in step_logs]),
            }, step=step)
            
            # Generate and log visualization if we have enough data points
            if len(self.router_logs) >= 5 * self.num_examples:
                self._log_visualizations_to_wandb(step)
    
    def _log_visualizations_to_wandb(self, step):
        """Create and log visualizations to wandb."""
        if wandb.run is None:
            return
            
        # Convert logs to DataFrame for easier processing
        logs_df = pd.DataFrame([
            {
                'step': log['step'],
                'example_id': log['example_id'],
                'entropy': log['entropy'],
                **{f'lora_{i}': p for i, p in enumerate(log['probs'])}
            }
            for log in self.router_logs
        ])
        
        # Only process if we have data
        if logs_df.empty:
            return
            
        try:
            # Create entropy over time plot
            plt.figure(figsize=(10, 6))
            for ex_id in logs_df['example_id'].unique():
                ex_df = logs_df[logs_df['example_id'] == ex_id]
                plt.plot(ex_df['step'], ex_df['entropy'], marker='o', label=f'Example {ex_id}')
            
            plt.title('Router Entropy Over Training')
            plt.xlabel('Training Steps')
            plt.ylabel('Entropy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save and log to wandb
            entropy_plot_path = os.path.join(self.log_dir, f'entropy_step_{step}.png')
            plt.savefig(entropy_plot_path)
            plt.close()
            
            wandb.log({
                "router/entropy_plot": wandb.Image(entropy_plot_path)
            }, step=step)
            
            # Create LoRA probability distribution plot (only if multiple LoRAs)
            lora_cols = [col for col in logs_df.columns if col.startswith('lora_')]
            if len(lora_cols) > 1:
                num_loras = len(lora_cols)
                
                plt.figure(figsize=(12, 8))
                for ex_id in logs_df['example_id'].unique():
                    ex_df = logs_df[logs_df['example_id'] == ex_id]
                    
                    # Get the last step for this example
                    last_step = ex_df['step'].max()
                    last_row = ex_df[ex_df['step'] == last_step].iloc[0]
                    
                    # Get the probabilities
                    probs = [last_row[f'lora_{i}'] for i in range(num_loras)]
                    
                    plt.subplot(2, 5, ex_id + 1)
                    plt.bar(range(num_loras), probs)
                    plt.title(f'Example {ex_id}')
                    plt.xlabel('LoRA Expert')
                    plt.ylabel('Probability')
                
                plt.tight_layout()
                
                # Save and log to wandb
                dist_plot_path = os.path.join(self.log_dir, f'lora_dist_step_{step}.png')
                plt.savefig(dist_plot_path)
                plt.close()
                
                wandb.log({
                    "router/lora_distribution": wandb.Image(dist_plot_path)
                }, step=step)
            
            # Create table of final LoRA selections
            wandb.log({
                "router/selections_table": wandb.Table(
                    dataframe=logs_df[logs_df['step'] == logs_df['step'].max()]
                )
            }, step=step)
            
        except Exception as e:
            logging.warning(f"Error creating router visualizations: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Final logging at the end of training."""
        # Final visualization
        if len(self.router_logs) > 0:
            self._log_visualizations_to_wandb(state.global_step)
            
            # Save all logs as JSON
            with open(os.path.join(self.log_dir, 'router_logs.json'), 'w') as f:
                json.dump(self.router_logs, f, indent=2)
            
            # Log all data to wandb
            if wandb.run is not None:
                wandb.save(os.path.join(self.log_dir, "*"))

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
    def __init__(self, hidden_size, num_loras, temperature=1.0):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_loras),
        )
        self.cached_probs = None
        self.temperature = temperature

    def forward(self, x):
        pooled = x[:, 0]
        logits = self.router(pooled.to(self.router[0].weight.dtype))
        # Apply temperature scaling - higher temperature = more uniform distribution
        scaled_logits = logits / self.temperature
        self.cached_probs = torch.softmax(scaled_logits, dim=-1)[0]  # shape: [num_loras]
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

    def __init__(self, model, router_cutoff_layer=2, num_loras=4, lora_r=8, lora_alpha=16, temperature=1.0):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_layers = len(model.model.layers)
        self.router_cutoff = router_cutoff_layer

        self.freeze_base_model()  # Freeze base model

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total_trainable:,}")

        self.router = GlobalRouter(model.config.hidden_size, num_loras, temperature)
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

        _ = self.router(hidden)  # sets cached_probs

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
        # Setup wandb configuration
        # if self.wandb_project and self.wandb_entity:
        #     os.environ['WANDB_PROJECT'] = self.wandb_project
        #     os.environ['WANDB_ENTITY'] = self.wandb_entity
        pass

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
    wrapped_model = QwenRouterWrapper(
        model,
        router_cutoff_layer=21,
        num_loras=10,
        lora_r=128,
        lora_alpha=256,
        temperature=1.0
    )
    
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
    args.ddp_find_unused_parameters = False
    
    # Initialize WandB if project is specified
    # if config.wandb_project and config.wandb_entity:
    #     wandb.init(
    #         project=config.wandb_project,
    #         entity=config.wandb_entity,
    #         config=log_config,
    #         name=f"lora-router-{args.output_dir.split('/')[-1]}"
    #     )
    
    # Create router analysis callback
    router_callback = RouterAnalysisCallback(
        model=wrapped_model,
        tokenizer=tokenizer,
        dataset=dataset['train'],
        num_examples=10,
        log_freq=100,
        log_dir=os.path.join(args.output_dir, 'router_logs')
    )

    # Initialize trainer with our MoLoRA model
    trainer = trl.SFTTrainer(
        model=wrapped_model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if 'test' in dataset else dataset['train'],
        args=args,
        data_collator=collator,
    )
    
    # Add the router analysis callback
    trainer.add_callback(router_callback)

    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
