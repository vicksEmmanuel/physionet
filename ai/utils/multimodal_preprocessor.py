from typing import Dict
import numpy as np
import torch
from torch import nn, tensor
from utils.types import InputBatch, OutputBatch
from utils.preprocessor import Preprocessor

class MultimodalPreprocessor(nn.Module):
    def __init__(
            self, 
            vit: any = None,
            model: any=None,
            device: str = "cpu",
            max_position_embeddings: int = 2408,
            vit_dim: int = 768,
            tokenizer: any=None,
        ):
        super().__init__()
        self.tokenizer = tokenizer
        self.preprocessor = Preprocessor(tokenizer, model=model)
        self.max_position_embeddings = max_position_embeddings
        self.device = device

        # Initialize ViT model and processor
        self.vit = vit
        self.model = model
        self.llm_dim = max_position_embeddings

        print(f"ViT model: {self.vit}")

        # Projection layer to match text embedding dimensions if needed
        self.vit_dim = vit_dim

        print(f"vit_dim x llm_dim : {self.vit_dim} x {self.llm_dim}")
        self.image_projection = nn.Linear(self.vit_dim, self.llm_dim)
        
        # Save token ids
        self.frame_token_id = self.preprocessor.frame_token_id
        self.action_token_id = self.preprocessor.action_token_id
        self.bbx1_token_id = self.preprocessor.bbx1_token_id
        self.bby1_token_id = self.preprocessor.bby1_token_id
        self.bbx2_token_id = self.preprocessor.bbx2_token_id
        self.bby2_token_id = self.preprocessor.bby2_token_id
        self.label_token_id = self.preprocessor.label_token_id

    def process_batch(self, batch: InputBatch) -> OutputBatch:
        """
        Process a batch of prompts and frames with position information.
        Handles both frame-containing prompts and frame-less prompts.
        """
        batch_size = len(batch['prompt'])

        # Tokenize all prompts
        encoded = self.tokenizer(
            batch['prompt'],
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_position_embeddings
        )

        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        _, sequence_length = input_ids.shape

        # First create text embeddings for all tokens
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            text_embeddings = self.model.get_input_embeddings()(input_ids)

        final_embedding = text_embeddings.to(torch.bfloat16)

        # Create loss_mask as ones
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32).to(self.device)

        # Special token ids
        special_tokens_ids = [
            self.frame_token_id,
            self.action_token_id,
            self.bbx1_token_id,
            self.bby1_token_id,
            self.bbx2_token_id,
            self.bby2_token_id,
            self.label_token_id
        ]

        # Set loss_mask to 0 where input_ids are special tokens
        for special_id in special_tokens_ids:
            loss_mask[(input_ids == special_id)] = 0.0

        # Process each item in batch
        for batch_idx in range(batch_size):
            try:
                n_frames = batch['n_frames'][batch_idx] if 'n_frames' in batch else 0

                # Only process frames if they exist and count is positive
                if n_frames > 0 and 'frame' in batch and batch_idx < len(batch['frame']):
                    current_frames = batch['frame'][batch_idx]

                    # Ensure we have valid frames to process
                    if not isinstance(current_frames, (list, tuple)) or len(current_frames) == 0:
                        continue

                    # Convert frames to tensor
                    frames = []
                    for f in current_frames[:n_frames]:
                        if isinstance(f, np.ndarray):
                            frames.append(torch.from_numpy(f))
                        elif isinstance(f, torch.Tensor):
                            frames.append(f)

                    if not frames:  # Skip if no valid frames
                        continue

                    # Process frames through ViT
                    frames_tensor = torch.stack(frames).to(self.device).to(torch.bfloat16)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.vit(pixel_values=frames_tensor)
                        image_embeddings = outputs.last_hidden_state
                        frame_embeddings = self.image_projection(image_embeddings)
                        frame_embeddings = frame_embeddings / (self.llm_dim**0.5)

                    # Find and replace frame token positions using masking
                    frame_positions = (input_ids[batch_idx] == self.frame_token_id)
                    if frame_positions.sum() > 0:
                        n_positions = min(frame_positions.sum(), len(frame_embeddings))
                        final_embedding[batch_idx].masked_scatter_(frame_positions.unsqueeze(-1), frame_embeddings[:n_positions])

            except Exception as e:
                print(f"Error processing batch item {batch_idx}: {str(e)}")
                print(f"Batch structure for item {batch_idx}: {[(k, type(batch[k][batch_idx]) if batch_idx < len(batch[k]) else 'index error') for k in batch.keys()]}")
                continue

        # Calculate position ids
        position_ids = torch.arange(sequence_length, device=self.device)[None, :].expand(batch_size, -1)
        position_ids = position_ids * attention_mask

        return {
            'embeddings': final_embedding,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'input_ids': input_ids,
            'loss_mask': loss_mask, 
        }

    def to_device(self, batch, device):
        """Move batch data to specified device"""
        if isinstance(batch, dict):
            return {k: self.to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self.to_device(x, device) for x in batch]
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch


    def forward(self, batch: InputBatch) -> OutputBatch:
        """Forward pass through the multimodal processor."""
        return self.process_batch(batch)