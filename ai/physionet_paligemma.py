from typing import Dict, List
import cv2
import torch
from peft import get_peft_model, LoraConfig
from PIL import Image

from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from dataset.dataset import Dataset
from torchvision import transforms
import yaml
from models.physionet_model import PhysionetLightingModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import subprocess
import re
import os
from torchvision import transforms
from bert_score import score as bert_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tqdm


class PhysiotherapyPaligemmaConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.physionet = PhysiotherapyPaligemma(
            dataset_config=config["dataset"],
            model_config=config["model"],
        )

    def train(self):
        self.physionet.train()

    def eval(self, config_path):
        self.physionet.evaluate(config_path=config_path)

    def inference(self, config_path, video_path, output_video_path):
        return self.physionet.inference(config_path=config_path, video_path=video_path, output_video_path = output_video_path)

    def inference_processed(self, config_path, video_path, output_video_path):
        return self.physionet.inference_processed(config_path=config_path, video_path=video_path, output_video_path = output_video_path)



class PhysiotherapyPaligemma:
    def __init__(self, dataset_config, model_config):
        self.dataset_config = dataset_config
        self.model_config = model_config

        self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.llm_model = AutoModelForCausalLM.from_pretrained("gpt2")

    def get_dataloader(self, processor):
        config = self.dataset_config
        self.model = self.llm_model
        self.tokenizer = self.llm_tokenizer
        self.processor = processor

        datamodule = Dataset(
            train_csv_path=config.get("train_csv_path", "dataset/data/train.csv"),
            test_csv_path=config.get("test_csv_path", "dataset/data/test.csv"),
            val_csv_path=config.get("val_csv_path", "dataset/data/val.csv"),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            resolution=config.get("resolution", 224),
            train_val_split=config.get("train_val_split", 0.8),
            pin_memory=config.get("pin_memory", True),
            shuffle=config.get("shuffle", True),
            tokenizer=self.tokenizer,
            model=self.model,
            utilize_memory=config.get("utilize_memory", False),
            file_index=config.get("file_index", 0),
            use_paligemma=config.get("use_paligemma", False),
            processor=processor
        )

        datamodule = datamodule.get_dataloader()
        return datamodule

    def get_model(self, device):
        MODEL_ID = self.model_config["model_id"]
        USE_LORA = self.model_config["use_lora"]
        USE_QLORA = self.model_config["use_qlora"]
        FREEZE_VISION = self.model_config["freeze_vision"]
        DEVICE = device

        processor = PaliGemmaProcessor.from_pretrained(MODEL_ID)

        if USE_LORA or USE_QLORA:
            lora_config = LoraConfig(
                r=8,
                target_modules=[
                    "q_proj",
                    "o_proj",
                    "k_proj",
                    "v_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                task_type="CAUSAL_LM",
            )
            if USE_QLORA:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_type=torch.bfloat16,
                )
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                quantization_config=bnb_config if USE_QLORA else None,
                torch_dtype=torch.bfloat16,
            )
            model = get_peft_model(model, lora_config)
            model = model.to(DEVICE)
            model.print_trainable_parameters()
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                MODEL_ID, device_map="auto"
            )

            if FREEZE_VISION:
                for param in model.vision_tower.parameters():
                    param.requires_grad = False

                for param in model.multi_modal_projector.parameters():
                    param.requires_grad = False

        TORCH_DTYPE = model.dtype

        return model, processor, TORCH_DTYPE


    def train(self):
        # Get the model, processor, and dtype
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model, processor, TORCH_DTYPE = self.get_model(device)

        # Get the dataloader
        datamodule = self.get_dataloader(processor)

        # Define the LightningModule
        lightning_model = PhysionetLightingModule(model, processor)

        # Define the Trainer
        wandb_logger = WandbLogger(project="physionet", name="physionet")
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.model_config["output_dir"],
            filename="paligemma-{epoch}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=self.model_config["num_epochs"],
            logger=wandb_logger,
            callbacks=[checkpoint_callback],
            precision="bf16" if self.model_config["fp16"] else 32,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            # strategy="ddp" if torch.cuda.device_count() > 1 else "dp",
        )

        if os.path.exists(self.model_config["resume_from_checkpoint"]):
            lightning_model = PhysionetLightingModule.load_from_checkpoint(self.model_config["resume_from_checkpoint"], model=model, processor=processor)
            print(f"Resuming from checkpoint: {self.model_config['resume_from_checkpoint']}")

        # if os.path.exists(self.model_config["resume_from_checkpoint"]):
        #     trainer.fit(model=lightning_model, datamodule=datamodule, ckpt_path=self.model_config["resume_from_checkpoint"])
        # else:
        trainer.fit(model=lightning_model, datamodule=datamodule)


    
        # Save the final model and processor
        model.save_pretrained(f"{self.model_config['output_dir']}/fine_tuned_paligemma")
        processor.save_pretrained(f"{self.model_config['output_dir']}/fine_tuned_paligemma")

        datamodule.setup(stage="test")
        trainer.test(model=lightning_model, datamodule=datamodule)


    def evaluate(self, config_path: str):
        # Get the model, processor, and dtype

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config = config["model"]
        # Set up model
        model_path = config.get("output_dir")
        if not model_path:
            raise ValueError("Model path not specified in config")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = f"{model_path}/fine_tuned_paligemma"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)

        print(f"Using device: {device}")
        if self.model_config["resume_from_checkpoint"] and self.model_config["resume_from_checkpoint"] != "none":
            lightning_model = PhysionetLightingModule.load_from_checkpoint(
                self.model_config["resume_from_checkpoint"], model=model, processor=processor
            )
            print(f"Resuming from checkpoint: {self.model_config['resume_from_checkpoint']}")
            processor = lightning_model.processor
            model = lightning_model.model

        # Get the dataloader
        datamodule = self.get_dataloader(processor)
        datamodule.setup(stage="fit")
        train_loader = datamodule.train_dataloader()

        datamodule.setup(stage="test")
        test_loader = datamodule.test_dataloader()

        def eval(dataloader, stage="train"):
            model.eval()
            total_instances = 0
            all_preds = []
            all_labels = []
            total_bert_f1 = 0.0
            total_detection_accuracy = 0.0

            with torch.no_grad():
                for batch in tqdm.tqdm(dataloader):
                    image = batch["image"]
                    prefix = batch["prefix"]
                    suffix = batch["suffix"]

                    batch_size = len(image)
                    total_instances += batch_size

                    for i in range(batch_size):
                        image_tensor = image[i]
                        prefix_value = prefix[i]
                        suffix_value = suffix[i]

                        prompts = "<image> detect"

                        # Process inputs with proper padding and truncation
                        inputs = processor(
                            text=[prompts],
                            images=[image_tensor],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=512,
                            do_rescale=False  # Since we already handled the rescaling
                        ).to(device)

                        output = model.generate(
                            **inputs,
                            temperature=0.7,  # Lower temperature makes output more focused/deterministic
                            top_p=0.9,  # Nucleus sampling threshold - only consider tokens with cumulative probability > 0.9
                            do_sample=True,  # Enable sampling (required for temperature and top_p to take effect)
                            max_new_tokens=20
                        )

                        decoded_output = processor.decode(output[0], skip_special_tokens=True)

                        # Case 1: If prefix_value contains "times", it's a summary/count-based evaluation
                        if "times" in prefix_value:
                            value_label = str(decoded_output).strip()
                            P, R, F1 = bert_score([value_label], [suffix_value], lang="en", verbose=False)
                            total_bert_f1 += F1.item()
                            all_preds.append(value_label)
                            all_labels.append(suffix_value)

                        # Case 2: If prefix_value does not contain "times", it's a detection/localization task
                        else:
                            # Extract predicted bounding boxes and actions
                            predicted_boxes_actions = self.extract_locators_and_actions(decoded_output)
                            ground_truth_boxes_actions = self.extract_locators_and_actions(suffix_value)

                            # Calculate detection accuracy (e.g., IoU for bounding boxes and action matching)
                            detection_accuracy = self.calculate_detection_accuracy(
                                predicted_boxes_actions, ground_truth_boxes_actions
                            )
                            total_detection_accuracy += detection_accuracy

                # Calculate average metrics
                avg_bert_f1 = total_bert_f1 / total_instances if "times" in prefix_value else 0.0
                avg_detection_accuracy = total_detection_accuracy / total_instances if "times" not in prefix_value else 0.0

                print(f"{stage} Evaluation Metrics:")
                if "times" in prefix_value:
                    print(f"BERT F1: {avg_bert_f1:.4f}")
                else:
                    print(f"Detection Accuracy: {avg_detection_accuracy:.4f}")

                # Log additional metrics if needed
                if all_preds and all_labels:
                    accuracy = accuracy_score(all_labels, all_preds)
                    precision = precision_score(all_labels, all_preds, average='macro')
                    recall = recall_score(all_labels, all_preds, average='macro')
                    f1 = f1_score(all_labels, all_preds, average='macro')
                    print(f"Accuracy: {accuracy:.4f}")
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1 Score: {f1:.4f}")

        eval(dataloader=train_loader, stage="train")
        eval(dataloader=test_loader, stage="test")



    def inference(self, config_path: str, video_path: str, output_video_path: str):
        """
        Perform inference on a video using a trained multimodal model.
        
        Args:
            config_path (str): Path to the configuration YAML file
            video_path (str): Path to the input video file
            output_video_path (str): Path to save the output video with bounding boxes
        
        Returns:
            list: List of model outputs for each frame
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config = config["model"]
        # Set up model
        model_path = config.get("inference_model")
        if not model_path:
            raise ValueError("Model path not specified in config")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = f"{model_path}"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(device)  # Move model to device
        processor = AutoProcessor.from_pretrained(model_id)

        # Video setup
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open video file {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.model_config["resume_from_checkpoint"] and self.model_config["resume_from_checkpoint"] != "none":
            lightning_model = PhysionetLightingModule.load_from_checkpoint(
                self.model_config["resume_from_checkpoint"], model=model, processor=processor
            )
            print(f"Resuming from checkpoint: {self.model_config['resume_from_checkpoint']}")
            processor = lightning_model.processor
            model = lightning_model.model

        # List to store frames with bounding boxes
        frames_with_boxes = []
        frame_count = 0

        # Dictionary to store action counts
        action_counts = {}
        # List to store frame-by-frame actions for detailed analysis
        frame_actions = []

        # Counter to skip frames
        skip_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames until the 30th frame
            if skip_counter < 30:
                skip_counter += 1
                continue

            # Reset skip counter after processing the 30th frame
            skip_counter = 0

            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            tensor_frame = pil_frame

            drawing_frame = frame_rgb

            list_of_actions = [
                "pelvis check", "demonstrating to the patient", "accessing the back of the patient",
                "foot examination", "patient taking off clothes", "use sanitizer", "look at the patient",
                "spine examination", "touching the back", "schrober examination", "knee examination",
                "patient sitting down", "lumbar extension", "lumbar flexion", "shoulder examination",
                "hands examination", "walking", "patient standing up"
            ]

            prompts = "<image> " + " ".join([f"detect {i};" for i in list_of_actions])

            # Process inputs with proper padding and truncation
            inputs = processor(
                text=[prompts],
                images=[tensor_frame],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                do_rescale=False  # Since we already handled the rescaling
            ).to(device)  # Move inputs to device

            output = model.generate(
                **inputs,
                temperature=0.2,  # Lower temperature makes output more focused/deterministic
                top_p=0.7,  # Nucleus sampling threshold - only consider tokens with cumulative probability > 0.9
                do_sample=True,  # Enable sampling (required for temperature and top_p to take effect)
                max_new_tokens=20
            )

            decoded_output = processor.decode(output[0], skip_special_tokens=True)
            val = self.extract_locators_and_actions(decoded_output)

            # Store actions for this frame
            current_frame_actions = []

            # Debugging: Print frame number and model output
            print(f"Frame {frame_count}: {decoded_output}")

            # Draw bounding boxes and action labels on the frame
            for locator, action in val:
                # Update action counts
                action_counts[action] = action_counts.get(action, 0) + 1
                current_frame_actions.append(action)

                x1, y1, x2, y2 = self.get_bounding_box_from_locator(locator, width, height)

                # Draw rectangle using x1,y1,x2,y2 coordinates
                cv2.rectangle(drawing_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Put text above the box
                cv2.putText(drawing_frame, action, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Store actions for this frame
            frame_actions.append({
                'frame_number': frame_count,
                'actions': current_frame_actions
            })

            # Append the frame with bounding boxes to the list
            frames_with_boxes.append(drawing_frame)

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        # Use FFmpeg to create the output video from the frames
        self.create_video_with_ffmpeg(frames_with_boxes, output_video_path, fps, width, height)

        # Generate action summary
        action_summary = " ".join([f"{action} {count} times" for action, count in action_counts.items()])
        dummy_frame = torch.zeros((3, 224, 224), dtype=torch.float32).to(device)  # Move dummy frame to device

        print(f"action summary {action_summary}")
        
        # Create detailed summary dictionary
        summary = {
            'total_frames': frame_count,
            'action_counts': action_counts,
            'frame_by_frame': frame_actions,
            'summary_text': action_summary
        }

        inputs = processor(
            text=[action_summary + " Answer:"],
            images=[dummy_frame],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            do_rescale=False  # Since we already handled the rescaling
        ).to(device)  # Move inputs to device

        output = model.generate(
            **inputs,
            temperature=0.7,  # Lower temperature makes output more focused/deterministic
            top_p=0.9,  # Nucleus sampling threshold - only consider tokens with cumulative probability > 0.9
            do_sample=True,  # Enable sampling (required for temperature and top_p to take effect)
            max_new_tokens=10
        )

        decoded_output = processor.decode(output[0], skip_special_tokens=True)

        with open(f'{output_video_path}.txt', 'w') as f:
            f.write(video_path + " ======>>> "+ str(decoded_output).replace(action_summary, ""))

        return {
            "text": str(decoded_output).replace(action_summary, ""),
            "video_path": output_video_path
        }

    def inference_processed(self, config_path: str, video_path: str, output_video_path: str):
        """
        Perform inference on a video using a trained multimodal model.
        
        Args:
            config_path (str): Path to the configuration YAML file
            video_path (str): Path to the input video file
            output_video_path (str): Path to save the output video with bounding boxes
        
        Returns:
            list: List of model outputs for each frame
        """
        
        # Get the directory containing output_video_path
        import os
        output_dir = os.path.dirname(output_video_path)
        
        # Walk through all files in the directory
        for root, dirs, files in os.walk(output_dir):
            # Filter only txt files
            txt_files = [f for f in files if f.endswith('.txt')]
            
            for txt_file in txt_files:
                txt_path = os.path.join(root, txt_file)
                
                try:
                    # Read content of each txt file
                    with open(txt_path, 'r') as f:
                        content = f.read().strip()
                    
                    splitter = " ======>>> "
                    # Split by the delimiter and take first part
                    if splitter in content:
                        stored_path = content.split(splitter, 1)[0].strip()
                        result = content.split(splitter, 1)[1].strip()
                        
                        # Compare with current output_video_path
                        if stored_path == video_path:
                            # Found matching path, write to output_video_path.txt
                           
                                
                            return {
                                "text": result,
                                "video_path": txt_path.replace("txt","")
                            }
                except Exception as e:
                    print(f"Error reading file {txt_path}: {str(e)}")
                    continue

        return {
            "text": "",
            "video_path": ""
        }


    def calculate_detection_accuracy(self, predicted_boxes_actions, ground_truth_boxes_actions):
        """
        Calculate detection accuracy by comparing predicted and ground truth bounding boxes and actions.
        
        Args:
            predicted_boxes_actions (list): List of tuples (predicted_box, predicted_action).
            ground_truth_boxes_actions (list): List of tuples (ground_truth_box, ground_truth_action).
        
        Returns:
            float: Detection accuracy (e.g., IoU for bounding boxes and action matching).
        """
        if not predicted_boxes_actions or not ground_truth_boxes_actions:
            return 0.0

        correct_matches = 0
        for pred_box, pred_action in predicted_boxes_actions:
            for gt_box, gt_action in ground_truth_boxes_actions:
                # Calculate IoU for bounding boxes
                iou = self.calculate_iou(pred_box, gt_box)
                # Check if actions match and IoU is above a threshold (e.g., 0.5)
                if pred_action == gt_action and iou >= 0.5:
                    correct_matches += 1
                    break

        accuracy = correct_matches / len(ground_truth_boxes_actions)
        return accuracy

    def calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            box1 (tuple): (x1, y1, x2, y2) coordinates of the first box.
            box2 (tuple): (x1, y1, x2, y2) coordinates of the second box.
        
        Returns:
            float: IoU value.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0 

    def extract_locators_and_actions(self, output):
        """
        Extract locators and action names from the given output string.
        
        Args:
            output (str): The output string containing locators and actions in the format:
                        <loc0063><loc0007><loc0113><loc0216> action
        
        Returns:
            list: A list of tuples where each tuple contains (coordinates, action)
                coordinates is a tuple of 4 integers (x1, y1, x2, y2)
        """
        # Pattern to match groups of 4 locators followed by an action
        pattern = r'(?:<loc(\d+)>){4}\s*([^<]+)'
        matches = re.finditer(pattern, output)
        
        extracted_data = []
        for match in matches:
            # Get all the numbers from the previous 4 loc tags
            coords = []
            start_pos = match.start()
            coord_pattern = r'<loc(\d+)>'
            coord_matches = re.finditer(coord_pattern, output[start_pos:match.end()])
            
            for coord_match in coord_matches:
                coords.append(int(coord_match.group(1)))
                
            if len(coords) == 4:  # Ensure we have all 4 coordinates
                action = match.group(2).strip()
                if action:  # Only add if we have an action
                    extracted_data.append((tuple(coords), action))
        
        return extracted_data
                
    def get_bounding_box_from_locator(self, coords, frame_width, frame_height):
        """
        Convert the 4-coordinate locator format to original frame coordinates.
        The input coordinates are in 224x224 space and need to be scaled back
        to the original frame dimensions.
        
        Args:
            coords (tuple): Tuple of 4 integers (x1, y1, x2, y2) in 224x224 space
            frame_width (int): Original frame width
            frame_height (int): Original frame height
        
        Returns:
            tuple: (x1, y1, x2, y2) scaled to original frame dimensions
        """
        x1, y1, x2, y2 = coords
        
        # Calculate scaling factors (original/224)
        scale_width = frame_width / 224
        scale_height = frame_height / 224
        
        # Scale back to original dimensions
        x1 = int(x1 * scale_width)
        y1 = int(y1 * scale_height)
        x2 = int(x2 * scale_width)
        y2 = int(y2 * scale_height)
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, min(x1, frame_width))
        y1 = max(0, min(y1, frame_height))
        x2 = max(0, min(x2, frame_width))
        y2 = max(0, min(y2, frame_height))
        
        return (x1, y1, x2, y2)

    def create_video_with_ffmpeg(self, frames, output_video_path, fps, width, height):
        """
        Create a video from a list of frames using FFmpeg.
        
        Args:
            frames (list): List of frames with bounding boxes.
            output_video_path (str): Path to save the output video.
            fps (int): Frames per second of the output video.
            width (int): Frame width.
            height (int): Frame height.
        """
        # Create a temporary directory to store the frames
        temp_dir = "temp_frames"
        os.makedirs(temp_dir, exist_ok=True)

        # Save each frame as an image in the temporary directory
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)

        # Use FFmpeg to create the video from the frames
        ffmpeg_command = [
            "ffmpeg",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={width}:{height}",
            output_video_path
        ]

        subprocess.run(ffmpeg_command)

        # Clean up the temporary directory
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)