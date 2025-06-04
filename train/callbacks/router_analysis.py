import os
import math
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import wandb
import transformers

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
            example = dataset[int(idx)]
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
        # Get device from model parameters instead of directly from model
        device = next(self.model.parameters()).device
        input_ids = input_dict['input_ids'].to(device)
        
        # Extract hidden states for router
        with torch.no_grad():
            # Process through model until router cutoff
            hidden = self.model.model.model.embed_tokens(input_ids)
            
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            
            # Import expand_attention_mask from the models module
            try:
                # When imported as package
                from ..models.lora import expand_attention_mask
            except ImportError:
                # When running directly
                from models.lora import expand_attention_mask
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
        
        temperature = getattr(self.model.router, 'temperature', 1.0)
        
        # Get router probabilities for each example
        for ex_id, inputs in enumerate(self.example_inputs):
            try:
                probs = self.get_router_probs(inputs)
                
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
            diversity_loss = getattr(self.model.router, 'last_diversity_loss', 0.0)
            expert_dropout = getattr(self.model.router, 'expert_dropout', 0.0)
            top_k = getattr(self.model.router, 'top_k', 1)
            
            wandb.log({
                "router/temperature": temperature,
                "router/avg_entropy": np.mean([log['entropy'] for log in step_logs]) if step_logs else 0.0,
                "router/diversity_loss": diversity_loss,
                "router/expert_dropout": expert_dropout,
                "router/top_k": top_k
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
