import numpy as np
from utils.types import DatasetItem, PreprocessedDatasetItem, PreprocessedPaligemmaDatasetItem
import torch
from PIL import Image
from torchvision import transforms


class Preprocessor:

    FRAME_TOKEN = "<frame>"
    ACTION_TOKEN = "<action>"
    BBX1_TOKEN = "<loc_x1>"
    BBY1_TOKEN = "<loc_y1>"
    BBX2_TOKEN = "<loc_x2>"
    BBY2_TOKEN = "<loc_y2>"
    LABEL_TOKEN = "<label>"

    def __init__(self, tokenizer, model, max_frames: int = 5000, processor=None ):
        self.tokenizer = tokenizer
        self.model = model

        # Add special tokens to tokenizer
        self.special_tokens = [
            self.FRAME_TOKEN, 
            self.ACTION_TOKEN, 
            self.BBX1_TOKEN, 
            self.BBY1_TOKEN, 
            self.BBX2_TOKEN, 
            self.BBY2_TOKEN, 
            self.LABEL_TOKEN
        ]

        self.max_frames = max_frames  # Maximum number of frames to pad to

        tokens_to_add = []
        for token in self.special_tokens:
            # Check if token is already in tokenizer's vocabulary
            if token not in self.tokenizer.get_vocab():
                tokens_to_add.append(token)
        
        # Only add new tokens if there are any
        if tokens_to_add:
            special_tokens_dict = {'additional_special_tokens': tokens_to_add}
            num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_tokens} new special tokens to the tokenizer")
            model.resize_token_embeddings(len(tokenizer))
        else:
            # print("All special tokens already present in tokenizer")
            pass
            
        
        print('Converting and saving token ids')
        self.frame_token_id = self.tokenizer.convert_tokens_to_ids(self.FRAME_TOKEN)
        self.action_token_id = self.tokenizer.convert_tokens_to_ids(self.ACTION_TOKEN)
        self.bbx1_token_id = self.tokenizer.convert_tokens_to_ids(self.BBX1_TOKEN)
        self.bby1_token_id = self.tokenizer.convert_tokens_to_ids(self.BBY1_TOKEN)
        self.bbx2_token_id = self.tokenizer.convert_tokens_to_ids(self.BBX2_TOKEN)
        self.bby2_token_id = self.tokenizer.convert_tokens_to_ids(self.BBY2_TOKEN)
        self.label_token_id = self.tokenizer.convert_tokens_to_ids(self.LABEL_TOKEN)
        self.processor = processor

    def create_prompt(self, data: DatasetItem) -> list[PreprocessedDatasetItem]:
        prompts = []
        dummy_frame = np.zeros((3, 224, 224), dtype=np.uint8)
        
        # Step 1: Process each frame individually with its annotations and create sequence prompts
        for seq in data["data"]:
            # Process individual frames with bounding boxes
            for idx, frame in enumerate(seq["frames_data"]):
                frame_prompt = ""
                frame_boxes = seq["frame_boxes"][idx]
                frame_actions = seq["frame_object_ids"][idx]
                
                frame_prompt += f"{self.FRAME_TOKEN} "
                for box, action in zip(frame_boxes, frame_actions):
                    frame_prompt += (
                        f"{self.BBX1_TOKEN}{box[0]}{self.BBX1_TOKEN} "
                        f"{self.BBY1_TOKEN}{box[1]}{self.BBY1_TOKEN} "
                        f"{self.BBX2_TOKEN}{box[2]}{self.BBX2_TOKEN} "
                        f"{self.BBY2_TOKEN}{box[3]}{self.BBY2_TOKEN} "
                        f"{self.ACTION_TOKEN}{action}{self.ACTION_TOKEN} "
                    )
                
                # Always create a list of exactly 100 frames
                frame_array = [np.array(frame)]  # Convert current frame to numpy array
                frame_array.extend([dummy_frame] * 99)  # Add 99 dummy frames
                
                prompts.append({
                    "prompt": frame_prompt + self.tokenizer.eos_token,
                    "label": "",
                    "frame": frame_array,  # Now consistently 100 frames
                    "n_frames": 1,
                    "size": data["size"],
                    "additional_info": data.get("additional_info", "")
                })
        
        # Create sequence-level prompt
        for seq in data["data"]:
            frame_prompt = ""
            frame_array = []
            
            for idx, frame in enumerate(seq["frames_data"][:100]):
                frame_prompt += f"{self.FRAME_TOKEN} "
                frame_array.append(np.array(frame))
            
            one_action = seq["frame_object_ids"][0][0]
            frame_prompt += f"{self.ACTION_TOKEN}{one_action}{self.ACTION_TOKEN} "
            
            # Ensure exactly 100 frames
            frame_array = frame_array[:100]  # Truncate if more than 100
            if len(frame_array) < 100:
                frame_array.extend([dummy_frame] * (100 - len(frame_array)))
            
            prompts.append({
                "prompt": frame_prompt + self.tokenizer.eos_token,
                "label": "",
                "frame": frame_array,  # Now consistently 100 frames
                "n_frames": min(len(seq["frames_data"]), 100),  # Cap at 100
                "size": data["size"],
                "additional_info": data.get("additional_info", "")
            })
        
        # Step 2: Create final prompt with action counts
        action_counts = {}
        for seq in data["data"]:
            for frame_actions in seq["frame_object_ids"]:
                for action in frame_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
        
        final_prompt = ""
        for action, count in action_counts.items():
            final_prompt += f"{self.ACTION_TOKEN}{action}{self.ACTION_TOKEN} {count} times "
        
        final_prompt += f"{self.LABEL_TOKEN}{data['value']}{self.LABEL_TOKEN}" + self.tokenizer.eos_token
        
        # Ensure final prompt also has exactly 100 frames
        dummy_frames = [dummy_frame] * 100
        
        prompts.append({
            "prompt": final_prompt,
            "label": data.get("value", ""),
            "frame": dummy_frames,
            "n_frames": 0,
            "size": data["size"],
            "additional_info": data.get("additional_info", "")
        })

        for i in range(-100, 0):
            print(f"\n=========================")
            print(f"{prompts[i]['prompt']}")
       
        
        return prompts


    def create_pali_gemma_prompt(self, data: DatasetItem) -> list[PreprocessedPaligemmaDatasetItem]:
        MAX_LENGTH = 512  # Maximum sequence length for inputs
        prompts = []
        dummy_frame = torch.zeros((3, 224, 224), dtype=torch.float32)

        for seq in data["data"]:
            for idx, frame in enumerate(seq["frames_data"]):
                frame_boxes = seq["frame_boxes"][idx]
                frame_actions = seq["frame_object_ids"][idx]
                
                width = seq["width"]
                height = seq["height"]

                # Get the original image size
                original_width = width
                original_height = height

                # Calculate the scaling factors for resizing
                scale_width = 224 / original_width
                scale_height = 224 / original_height

                suffix_components = []
                prefix_components = []
                for box, action in zip(frame_boxes, frame_actions):
                    # Resize bounding box coordinates
                    resized_box = [
                        int(box[0] * scale_width),  # x1
                        int(box[1] * scale_height),  # y1
                        int(box[2] * scale_width),  # x2
                        int(box[3] * scale_height),  # y2
                    ]

                    # Normalize and bin bounding box coordinates
                    loc_string = ""
                    for coord in resized_box:
                        cord_val = min(coord, 1023)
                        loc_string += f"<loc{cord_val:04d}>"
                    
                    # Combine into suffix component
                    suffix_component = f"{loc_string} {action}"
                    suffix_components.append(suffix_component)
                    prefix_components.append(action)
                
                # Combine suffix components into a single suffix
                suffix = " ; ".join(suffix_components)
                prefix = "detect " + " ; ".join(prefix_components)

                prompts.append({
                    "suffix": suffix,
                    "prefix": "<image>" + prefix,
                    "image": frame
                })

        # Process final summary prompt
        action_counts = {}
        for seq in data["data"]:
            for frame_actions in seq["frame_object_ids"]:
                for action in frame_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
        
        final_prompt = " ".join([f"{action} {count} times" for action, count in action_counts.items()])

        prompts.append({
            "suffix": data['value'],
            "prefix": "<image>"+ final_prompt,
            "image": dummy_frame
        })

        for i in range(len(prompts)):
            print(prompts[i])
            print(f"\n\n")
            print(f"=============>>>>")

        return prompts

    def extract_coordinates_of_bounding_boxes(self, input_string):
        # Split the input string by the tokens
        parts = input_string.split()
        
        # Initialize an empty list to hold the coordinates
        coordinates = []
        
        # Extract values by searching for the tokens
        for part in parts:
            if self.BBX1_TOKEN in part:
                x1_value = part.replace(self.BBX1_TOKEN, "").strip()
                coordinates.append(float(x1_value))
            elif self.BBY1_TOKEN in part:
                y1_value = part.replace(self.BBY1_TOKEN, "").strip()
                coordinates.append(float(y1_value))
            elif self.BBX2_TOKEN in part:
                x2_value = part.replace(self.BBX2_TOKEN, "").strip()
                coordinates.append(float(x2_value))
            elif self.BBY2_TOKEN in part:
                y2_value = part.replace(self.BBY2_TOKEN, "").strip()
                coordinates.append(float(y2_value))
        
        return coordinates
    
    def extract_action(self, input_string):
        # Split the input string by the tokens
        parts = input_string.split()
        
        # Initialize an empty list to hold the action
        action = []
        
        # Extract values by searching for the tokens
        for part in parts:
            if self.ACTION_TOKEN in part:
                action_value = part.replace(self.ACTION_TOKEN, "").strip()
                action.append(action_value)
        
        return action
    
    def extract_label(self, input_string):
        # Split the input string by the tokens
        parts = input_string.split()
        
        # Initialize an empty list to hold the label
        label = []
        
        # Extract values by searching for the tokens
        for part in parts:
            if self.LABEL_TOKEN in part:
                label_value = part.replace(self.LABEL_TOKEN, "").strip()
                label.append(label_value)
        
        return label




                