import pytorch_lightning as pl
import torch


class PhysionetLightingModule(pl.LightningModule):
    def __init__(self, model, processor, learning_rate=5e-5):
        super().__init__()
        self.model = model
        self.processor = processor
        self.learning_rate = learning_rate

        self.model = self.model.to(torch.float32)

    def forward(self, inputs):
        return self.model(**inputs)

    def _process_batch(self, batch):
        images = batch["image"] 
        prefixes = batch["prefix"] 
        suffixes = batch["suffix"] 

        images = [image for image in images]

        # Tokenize the inputs
        inputs = self.processor(
            text=prefixes,
            images=images,
            suffix=suffixes,
            return_tensors="pt",
            padding=True,
            do_rescale=False
        ).to(self.device)

        # Convert tensors to appropriate types
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                if key in ['input_ids', 'attention_mask', 'token_type_ids']:
                    inputs[key] = value.long()  # Index-related tensors should be long
                elif key == 'pixel_values':
                    inputs[key] = value.to(torch.bfloat16)  # Ensure pixel values are float32
                elif key == 'labels':
                    inputs[key] = value.long()  # Labels should also be long for classification
                else:
                    inputs[key] = value.to(torch.bfloat16)  # Other tensors as float32

        # Convert model to float32 if needed
        self.model.to(torch.bfloat16)

        return inputs

    def training_step(self, batch, batch_idx):
        inputs = self._process_batch(batch)
        
        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self._process_batch(batch)

        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss

        # Log the validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def testing_step(self, batch, batch_idx):
        inputs = self._process_batch(batch)
        
        # Forward pass
        outputs = self.model(**inputs)
        loss = outputs.loss

        # Log the training loss
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.testing_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer