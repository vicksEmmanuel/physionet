import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List
from utils.multimodal_preprocessor import MultimodalPreprocessor


class PhysiotherapyModel(nn.Module):
    def __init__(
            self, 
            vit: any = None,
            max_seq_length: int = 2048,
            device: str = "cpu",
            tokenizer: any=None,
            model: any = None,
            vit_config: any = None
        ):
        super().__init__()

        self.device = device

        # Load Unsloth optimized LLaMA model
        self.model = model
        self.tokenizer = tokenizer

        self.processor = MultimodalPreprocessor(
            vit=vit,
            model = self.model,
            device=self.device,
            max_position_embeddings=max_seq_length,
            tokenizer=tokenizer,
            vit_dim=vit_config['hidden_size']
        )

    def to_device(self, batch, device):
        """Move batch data to specified device"""
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self.to_device(x, device) for x in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    def forward(self, batch: Dict):
        output = self.processor(batch)
        output = self.to_device(output, self.device)

        model_output = self.model(
            inputs_embeds=output['embeddings'],
            attention_mask=output['attention_mask'],
            position_ids=output['position_ids']
        )

        return {
            "model_output": model_output,
            "processed_output": output 
        }