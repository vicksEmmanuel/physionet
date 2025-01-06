import os

import pandas as pd
from dataset.multimodal_dataloader_memory_mgt import MultimodalDataModuleMemoryManagement
from dataset.multimodal_dataloader import MultimodalDataModule


class Dataset:
    def __init__(self, train_csv_path, test_csv_path, val_csv_path, batch_size, num_workers, resolution, train_val_split, pin_memory, shuffle, tokenizer, model, utilize_memory, file_index, use_paligemma, processor):
        self.train_csv_path = train_csv_path
        self.test_csv_path = test_csv_path
        self.val_csv_path = val_csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.train_val_split = train_val_split
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.model = model
        self.utilize_memory = utilize_memory
        self.file_index = file_index
        self.use_paligemma = use_paligemma
        self.processor = processor



    def get_dataloader(self):
        if self.utilize_memory:
            print("Using memory management")
            return self.get_memory_dataloader()
        
        return self.get_all_dataloader()
    
    def get_all_dataloader(self):
        return  MultimodalDataModule(
            train_csv_path=self.train_csv_path,
            test_csv_path=self.test_csv_path,
            val_csv_path=self.val_csv_path,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            resolution=self.resolution,
            train_val_split=self.train_val_split,
            pin_memory=True,
            shuffle=False,
            tokenizer=self.tokenizer,
            model=self.model,
            use_paligemma=self.use_paligemma,
            processor=self.processor
        )
    
    def get_memory_dataloader(self):
        # Check if files exist
        if not os.path.exists(self.train_csv_path):
            raise FileNotFoundError(f"Training CSV file not found: {self.train_csv_path}")
        if self.val_csv_path and not os.path.exists(self.val_csv_path):
            raise FileNotFoundError(f"Validation CSV file not found: {self.val_csv_path}")
        if self.test_csv_path and not os.path.exists(self.test_csv_path):
            raise FileNotFoundError(f"Test CSV file not found: {self.test_csv_path}")

        # Load CSV files
        self.data = pd.read_csv(self.train_csv_path)
        self.val_data = pd.read_csv(self.val_csv_path) if self.val_csv_path else None
        self.test_data = pd.read_csv(self.test_csv_path) if self.test_csv_path else None

        # Check if file_index is valid for training data
        if self.file_index >= len(self.data):
            raise ValueError(f"Invalid file index {self.file_index} for training data")

        # Get the rows directly instead of using iterrows()
        train_row = self.data.iloc[self.file_index]
        
        # Handle validation data
        val_row = None

        print(f"val data {self.val_data}")

        if self.val_data is not None:
            if self.file_index >= len(self.val_data):
                self.val_data = self.val_data.sample(frac=1).reset_index(drop=True)
            val_row = self.val_data.iloc[self.file_index]

        # Handle test data
        test_row = None
        if self.test_data is not None:
            if self.file_index >= len(self.test_data):
                self.test_data = self.test_data.sample(frac=1).reset_index(drop=True)
            test_row = self.test_data.iloc[self.file_index]

        return MultimodalDataModuleMemoryManagement(
            train_row=train_row,
            val_row=val_row,
            test_row=test_row,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            tokenizer=self.tokenizer,
            model=self.model,
            resolution=self.resolution,
            train_val_split=self.train_val_split,
            use_paligemma=self.use_paligemma,
            processor=self.processor
        )