import os
from utils.types import InputBatch
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List
from torch.nn import CrossEntropyLoss

from models.physiotherapy_model import PhysiotherapyModel

class MultimodalTrainer(pl.LightningModule):
    def __init__(
        self,
        model: PhysiotherapyModel,
        tokenizer: any,
        learning_rate: float = 2e-5,
        max_epochs: int = 10,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model: PhysiotherapyModel = model
        
        # Use CrossEntropyLoss with label smoothing
        self.loss = CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,  # Ignore padding tokens
            label_smoothing=0.1  # Apply label smoothing for better generalization
        )
        
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    def forward(self, batch: Dict[str, torch.Tensor]):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)

        # Extract logits and labels
        logits = outputs["model_output"]["logits"]
        labels = outputs['processed_output']['input_ids']

        # Compute loss
        loss = self.loss(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        # Extract logits and labels
        logits = outputs["model_output"]["logits"]
        labels = outputs['processed_output']['input_ids']

        # Compute loss
        loss = self.loss(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.testing_step(batch=batch, batch_idx=batch_idx)

    def testing_step(self, batch, batch_idx):
        outputs = self(batch)

        # Extract logits and labels
        logits = outputs["model_output"]["logits"]
        labels = outputs['processed_output']['input_ids']

        # Compute loss
        loss = self.loss(
            logits.view(-1, logits.size(-1)), 
            labels.view(-1)
        )

        # Log metrics
        self.log('test_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        print(f"hparams {self.hparams}")

        # Create optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-5,
        )

        # Create scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def generate(
        self,
        batch: InputBatch,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """Generate text from preprocessed input."""

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Initial forward pass
            value = self.model(batch)
            outputs = value
            processed_output = value['processed_output']

            # Get initial inputs
            input_ids = processed_output['input_ids']
            attention_mask = processed_output['attention_mask']
            position_ids = processed_output['position_ids']
            inputs_embeds = processed_output['embeddings']

            # Initialize storage for generated tokens
            generated_tokens = []
            stop_token = self.model.tokenizer.eos_token_id

            # Generate tokens one by one
            for _ in range(max_new_tokens):
                # Get model outputs
                outputs = self.model.model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask
                )

                # Get logits for next token
                next_token_logits = outputs["logits"][:, -1, :]

                # Sample next token
                if do_sample:
                    # Apply temperature and softmax
                    next_token_logits = torch.softmax(next_token_logits / temperature, dim=-1)

                    # Apply top_p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                    # Sample from the filtered distribution
                    next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
                else:
                    # Greedy selection
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                next_token = next_token.squeeze(0)  # Remove batch dimension
                generated_tokens.append(next_token)

                # Stop if we hit the stop token
                if next_token.item() == stop_token:
                    break

                # Update inputs_embeds and attention_mask for next iteration
                next_token_embeds = self.model.model.get_input_embeddings()(next_token.unsqueeze(0))  # Get embeddings for the new token
                inputs_embeds = torch.cat([inputs_embeds, next_token_embeds], dim=1)  # Append new token embeddings
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=attention_mask.device)], dim=1)  # Update attention mask

                # Update position_ids
                position_ids = torch.arange(inputs_embeds.size(1), dtype=torch.long, device=inputs_embeds.device).unsqueeze(0)

            # Combine all generated tokens
            generated_tokens = torch.cat(generated_tokens, dim=-1)

            # Decode the generated tokens
            decoded = self.model.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True
            )

            return [decoded]