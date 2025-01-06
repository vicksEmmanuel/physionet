from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
from typing import Optional, Dict, Any



class DataItem(TypedDict):
    frames_data: List[Image.Image] # List of transformed frames (PIL Images)
    frame_indices: List[int] # List of frame numbers
    frame_boxes: List[List[List[float]]] # List of frames, each containing list of boxes [x1,y1,x2,y2]
    frame_object_ids: List[List[str]] # List of frames, each containing list of object IDs or action labels


class DatasetItem(TypedDict):
    size: Dict[str, int] # {width: int, height: int}
    value: str # Good, Bad, or Brief
    additional_info: str # Additional information
    data: List[DataItem] # List of sequence data



class PreprocessedDatasetItem(TypedDict):
    prompt: str
    label: str
    frame: List[np.ndarray]  # List of frames (numpy arrays)
    n_frames: int            # Number of actual frames before padding
    size: Dict[str, int]     # {width: int, height: int}
    additional_info: str


class PreprocessedPaligemmaDatasetItem(TypedDict):
    image: List[np.ndarray]  # List of frames (numpy arrays)
    prefix: str
    suffix: str


class InputBatch(TypedDict):
    prompt: List[str]                              # batch['prompt'] is used for tokenization
    n_frames: List[int]                            # batch['n_frames'][batch_idx] accessed 
    frame: List[List[Union[np.ndarray, torch.Tensor]]]  # batch['frame'][batch_idx] accessed and converted to tensor
    # Note: frames are either numpy arrays or torch tensors before processing

class OutputBatch(TypedDict):
    embeddings: torch.Tensor      # [batch_size, sequence_length, llm_dim]
    attention_mask: torch.Tensor  # [batch_size, sequence_length]
    position_ids: torch.Tensor    # [batch_size, sequence_length]
    input_ids: torch.Tensor      # [batch_size, sequence_length]