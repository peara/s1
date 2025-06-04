import os
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# Import our custom modules - handle both package and direct imports
try:
    # When running as a package (from parent directory)
    from train.models.lora import QwenRouterWrapper
    from train.callbacks.router_analysis import RouterAnalysisCallback
except ImportError:
    # When running directly from train directory
    from models.lora import QwenRouterWrapper
    from callbacks.router_analysis import RouterAnalysisCallback

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
        temperature=3.0,  # High temperature for more uniform initial distribution
        diversity_weight=0.5,  # Fairly strong diversity penalty to encourage uniform usage
        expert_dropout=0.3,  # Drop each expert with 30% probability during training
        top_k=2  # Select top 2 LoRAs for each input
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
    # No need, we use default wandb settings
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
        log_freq=5,
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
    # Note: commented out until we have a good result to save
    # trainer.save_model(output_dir=args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
