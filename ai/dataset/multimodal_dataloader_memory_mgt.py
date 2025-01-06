from dataset.multimodal_dataset_memory_mgt import MultimodalMemoryManagementDataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from typing import Optional, Dict, Any

class MultimodalDataModuleMemoryManagement(pl.LightningDataModule):
    def __init__(
        self,
        train_row,
        val_row,
        test_row,
        batch_size: int = 32,
        num_workers: int = 0,
        resolution: int = 224,
        train_val_split: float = 0.8,
        pin_memory: bool = True,
        shuffle: bool = True,
        normalization: tuple = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet normalization
        tokenizer=None,
        model=None,
        use_paligemma=False,
        processor=None
    ):
        """
        Args:
            val_row: Data
            train_row: Data
            test_row: Data
            batch_size: Batch size for all dataloaders
            num_workers: Number of workers for all dataloaders
            resolution: Image resolution to resize to
            train_val_split: Fraction of data to use for training when val_csv_path is None
            pin_memory: Whether to pin memory in dataloaders
            shuffle: Whether to shuffle training data
            normalization: Tuple of (means, stds) for normalization
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.normalization = normalization
        self.tokenizer = tokenizer
        self.model = model
        self.train_row = train_row
        self.val_row = val_row
        self.test_row = test_row
        self.use_paligemma = use_paligemma
        self.processor = processor



    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage of training"""
        print(f"Setting up Data")
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to a fixed size
            transforms.ToTensor(),          # Convert image to tensor
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to a fixed size
            transforms.ToTensor(),          # Convert image to tensor
        ])

        print(f"Stage: {stage}")

        if stage == "fit" or stage is None:
            # Setup training dataset
            self.train_dataset = MultimodalMemoryManagementDataset(
                row=self.train_row,
                transform=self.train_transform ,
                tokenizer=self.tokenizer,
                model=self.model,
                use_paligemma=self.use_paligemma,
                processor=self.processor
            )

            self.val_dataset = self.train_dataset


        if stage == "test" or stage is None:
            self.test_dataset = MultimodalMemoryManagementDataset(
                row=self.test_row,
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                model=self.model,
                use_paligemma=self.use_paligemma,
                processor=self.processor
            )

    def train_dataloader(self) -> DataLoader:
        """Create the training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create the test dataloader if test_csv_path was provided"""
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return None

    def get_transform_params(self) -> Dict[str, Any]:
        """Get the transformation parameters used"""
        return {
            'resolution': self.resolution,
            'normalization': self.normalization
        }





