import os
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_from_disk
import transformers
import trl
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.modeling_utils import PreTrainedModel


class MoLoRALinear(nn.Module):
    """Linear layer with multiple LoRA adapters and routing"""
    def __init__(self, base_linear, num_adapters=10, r=16, lora_alpha=16):
        super().__init__()
        self.base_linear = base_linear
        self.num_adapters = num_adapters
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        # Original dimensions
        in_features, out_features = base_linear.in_features, base_linear.out_features
        
        # Freeze base linear layer parameters
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # Create multiple LoRA adapters (A and B matrices for each adapter)
        self.lora_A = nn.Parameter(torch.zeros(num_adapters, r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(num_adapters, out_features, r))
        
        # Initialize LoRA weights
        for i in range(num_adapters):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])
            
        # Router - simple MLP
        router_dim = 128
        self.router = nn.Sequential(
            nn.Linear(in_features, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, num_adapters)
        )
    
    def forward(self, x):
        # Apply base layer
        base_output = self.base_linear(x)
        
        # Get batch size and sequence length
        batch_size, seq_len, hidden_size = x.shape
        
        # Get router weights (batch_size x num_adapters)
        router_input = x[:, 0]  # Use first token for routing
        adapter_weights = F.softmax(self.router(router_input), dim=-1)  # (batch_size, num_adapters)
        
        # Initialize lora output
        lora_output = torch.zeros_like(base_output)
        
        # Apply each adapter and weight by router predictions
        for i in range(self.num_adapters):
            # Reshape adapter weights for broadcasting
            weights_for_adapter = adapter_weights[:, i].view(batch_size, 1, 1)
            
            # Get LoRA contribution for this adapter
            adapter_output = (x @ self.lora_A[i].T @ self.lora_B[i].T) * self.scaling
            
            # Weight adapter output by router prediction
            weighted_adapter = adapter_output * weights_for_adapter
            
            # Add to total LoRA output
            lora_output += weighted_adapter
        
        # Combine base output with weighted LoRA output
        return base_output + lora_output


class MoLoRAModel(PreTrainedModel):
    """Base class for MoLoRA models - inherits PreTrainedModel for full compatibility"""
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, num_adapters=10, target_modules=None, r=16, lora_alpha=16, *args, **kwargs):
        # Load the base model using standard from_pretrained
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        
        # Create and return MoLoRA-enabled model
        molora_model = cls(base_model.config)
        molora_model.init_molora(base_model, num_adapters, target_modules, r, lora_alpha)
        return molora_model
    
    def __init__(self, config):
        # ignore num_adapters as it is not a standard config parameter
        # and is handled in init_molora
        super().__init__(config)
        # Initialize basic properties but don't set up layers yet
        # This happens in init_molora after the base model is loaded
    
    def init_molora(self, pretrained_model, num_adapters, target_modules, r, lora_alpha):
        """Initialize MoLoRA with a base model"""
        self.pretrained_model = pretrained_model
        self.num_adapters = num_adapters
        self.target_modules = target_modules
        self.r = r
        self.lora_alpha = lora_alpha
        
        # Freeze all base model parameters
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
            
        # Replace target modules with MoLoRA versions
        self._replace_modules()
    
    def _replace_modules(self):
        """Replace target linear layers with MoLoRA versions"""
        # Track replaced modules
        self.molora_layers = {}

        # Find and replace target modules
        for name, module in self.pretrained_model.named_modules():
            if any(target in name for target in self.target_modules) and isinstance(module, nn.Linear):
                # Get parent module
                parent_name, child_name = name.rsplit(".", 1)
                parent = self.pretrained_model

                parts = parent_name.split('.')
                for part in parts:
                    parent = getattr(parent, part)
                
                # Replace with MoLoRA version
                molora_layer = MoLoRALinear(
                    module, 
                    num_adapters=self.num_adapters,
                    r=self.r,
                    lora_alpha=self.lora_alpha
                )
                setattr(parent, child_name, molora_layer)
                self.molora_layers[name] = molora_layer
                
                logging.info(f"Replaced {name} with MoLoRA layer")
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Pass through to base model with explicit parameters
        return self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """Pass through to base model's prepare_inputs_for_generation"""
        return self.pretrained_model.prepare_inputs_for_generation(*args, **kwargs)
   
    def get_router_weights(self):
        """Get router weights for analysis"""
        weights = {}
        for name, layer in self.molora_layers.items():
            weights[name] = {
                'A': layer.lora_A.data.clone(),
                'B': layer.lora_B.data.clone(),
                'router': layer.router[0].weight.data.clone(),
            }
        return weights

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

    # Create MoLoRA model with routing using from_pretrained pattern
    model = MoLoRAModel.from_pretrained(
        config.model_name,
        num_adapters=2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=16,
        lora_alpha=16,
        device_map="auto" if "70B" in config.model_name else None,
        torch_dtype="auto" if "70B" in config.model_name else None,
        attn_implementation="flash_attention_2" if "70B" in config.model_name else None,
        use_cache=False if "70B" in config.model_name else None
    )
    
    # Count trainable parameters
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logging.info(f"Trainable parameters: {trainable_params:,d} ({trainable_params/all_params:.2%} of all parameters)")
    
    # Load dataset
    dataset = load_from_disk(config.train_file_path)

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
        model,
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