from typing import Dict, List, Optional, TypedDict, Union

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import json
import cv2
import os
import torch
from torchvision import transforms
from PIL import Image
import os
from typing import Dict

from utils.types import PreprocessedDatasetItem
from utils.preprocessor import  Preprocessor


"""
    MultimodalMemoryManagementDataset class takes a single video row file with columns 'file_path', 'annotation_path', 'value' and 'additional_info' as input.
"""
class MultimodalMemoryManagementDataset(Dataset):
    def __init__(self, row, transform=None, tokenizer=None, model=None, use_paligemma=False, processor=None):
        """
        Args:
            row (dict): Contains dict as 'file_path' and 'annotation_path'
            transform (callable, optional): Optional transform to be applied on video frames
        """
        self.row = row
        self.transform = transform
        self.use_paligemma = use_paligemma
        self.processor = processor
        self.dataset_value: List[PreprocessedDatasetItem] = self.__load__(tokenizer, model)


    def __load__(self, tokenizer, model) :
        all = []
        row = self.row
        # read json file
        with open(row['annotation_path']) as f:
            json_data = json.load(f)

        all_annotated_frames = self._group_sequences(json_data["frames"])

        cap = cv2.VideoCapture(row['file_path'])
        if not cap.isOpened():
            print(f"Error: Could not open video file {row['file_path']}")
            return []
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            data = []
            for annotated_frame in all_annotated_frames:
                start_idx = annotated_frame['start_frame']
                end_idx = annotated_frame['end_frame']

                if start_idx >= total_frames or end_idx >= total_frames:
                    print(f"Error: Invalid frame indices {start_idx}-{end_idx} for video with {total_frames} frames")
                    continue

                frames = []
                frame_indices = []
                frame_boxes = []
                frame_object_ids = []
                
                # Set position to start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                
                for current_idx in range(start_idx, end_idx + 1):
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Error: Failed to read frame {current_idx} from video")
                        break
                        
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image for torchvision transforms
                    frame = Image.fromarray(frame)
                    
                    # Apply transforms if provided
                    # if self.transform is not None:
                    #     frame = self.transform(frame)
                    
                    frames.append(frame)
                    frame_indices.append(current_idx)

                    # Process annotations for this frame
                    for _, figure in enumerate(annotated_frame['frames']):
                        if figure['index'] == current_idx:
                            boxes = []
                            object_ids = []
                            
                            for figure in figure['figures']:
                                exterior_points = figure['geometry']['points']['exterior']
                                x1, y1 = exterior_points[0]
                                x2, y2 = exterior_points[1]
                                
                                boxes.append([x1, y1, x2, y2])
                                object_ids.append(
                                    self.create_id_to_text(json_data['objects'], figure['objectId'])
                                )
                            
                            frame_boxes.append(boxes)
                            frame_object_ids.append(object_ids)

                data.append({
                    'frames_data': frames,
                    'frame_indices': frame_indices,
                    'frame_boxes': frame_boxes,
                    'frame_object_ids': frame_object_ids,
                    "width": width,
                    "height": height
                })

        finally:
            cap.release()
            all.append({
                "data": data,
                'additional_info': row.get("additional_info",""),
                'value': f"Answer: " + row.get("value",""),
                'size': json_data.get("size",""),
            })
        
        print(f"Done ===>>><<<===")


        prep = Preprocessor(tokenizer=tokenizer, model=model, processor=self.processor)
        all_prompts = []

        for data in all:
            if self.use_paligemma:
                values = prep.create_pali_gemma_prompt(data)
            else:
                values = prep.create_prompt(data)
            for value in values:
                all_prompts.append(value)

        # Invert the list
        all_prompts = all_prompts[::-1]
        return all_prompts

    def _group_sequences(self, json_data: List[Dict]) -> List[Dict]:
        """
        Group frames into sequences based on consecutive frame indices.
        A new sequence starts whenever there's a gap in frame indices.
        
        Args:
            json_data: Loaded JSON data containing frames and annotations
            
        Returns:
            List of sequence dictionaries containing grouped frames
        """
        # Sort frames by index
        frames = sorted(json_data, key=lambda x: x['index'])
        
        sequences = []
        current_sequence = []
        last_index = None
        
        for frame in frames:
            if last_index is None or frame['index'] == last_index + 1:
                current_sequence.append(frame)
            else:
                # Add the sequence regardless of length
                if current_sequence:  # Only add if there are frames
                    sequences.append({
                        'frames': current_sequence.copy(),
                        'start_frame': current_sequence[0]['index'],
                        'end_frame': current_sequence[-1]['index']
                    })
                current_sequence = [frame]
            last_index = frame['index']
        
        # Add the last sequence
        if current_sequence:  # Only add if there are frames
            sequences.append({
                'frames': current_sequence,
                'start_frame': current_sequence[0]['index'],
                'end_frame': current_sequence[-1]['index']
            })
                
        return sequences
    

    def create_id_to_text(self, objects_data: List[Dict], objectId: str) -> str: # type: ignore
        """ 
            Args: objects_data: List of objects data
                    objectId: Object id to convert to text
            Returns: Object text for the given objectId
        """
        objects_data = {obj['id']: obj['classTitle'] for obj in objects_data}
        return objects_data[objectId]


    def __len__(self):
        return len(self.dataset_value)

    def __getitem__(self, idx: int):
        value = self.dataset_value[idx]
        return value
    
