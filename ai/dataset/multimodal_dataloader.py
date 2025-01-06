from dataset.multimodal_dataset import MultimodalDataset
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

class MultimodalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        val_csv_path: Optional[str] = None,
        test_csv_path: Optional[str] = None,
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
            train_csv_path: Path to training CSV file
            val_csv_path: Optional path to validation CSV file
            test_csv_path: Optional path to test CSV file
            batch_size: Batch size for all dataloaders
            num_workers: Number of workers for all dataloaders
            resolution: Image resolution to resize to
            train_val_split: Fraction of data to use for training when val_csv_path is None
            pin_memory: Whether to pin memory in dataloaders
            shuffle: Whether to shuffle training data
            normalization: Tuple of (means, stds) for normalization
        """
        super().__init__()
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.test_csv_path = test_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.normalization = normalization
        self.tokenizer = tokenizer
        self.model = model
        self.use_paligemma = use_paligemma
        self.processor = processor

        # Validation of inputs
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training CSV file not found: {train_csv_path}")
        if val_csv_path and not os.path.exists(val_csv_path):
            raise FileNotFoundError(f"Validation CSV file not found: {val_csv_path}")
        if test_csv_path and not os.path.exists(test_csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {test_csv_path}")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage of training"""
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to a fixed size
            transforms.ToTensor(),          # Convert image to tensor
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize image to a fixed size
            transforms.ToTensor(),          # Convert image to tensor
        ])

        if stage == "fit" or stage is None:
            # Setup training dataset
            self.train_dataset = MultimodalDataset(
                csv_path=self.train_csv_path,
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                model=self.model,
                use_paligemma=self.use_paligemma,
                processor=self.processor
            )

            # Setup validation dataset
            if self.val_csv_path:
                self.val_dataset = MultimodalDataset(
                    csv_path=self.val_csv_path,
                    transform=self.val_transform,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    use_paligemma=self.use_paligemma,
                    processor=self.processor
                )
            else:
                # Split training data if no validation csv provided
                train_size = int(len(self.train_dataset) * self.train_val_split)
                val_size = len(self.train_dataset) - train_size
                self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset,
                    [train_size, val_size]
                )

        if stage == "test" or stage is None:
            if self.test_csv_path:
                self.test_dataset = MultimodalDataset(
                    csv_path=self.test_csv_path,
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





