import os
import traceback
import cv2
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb
import yaml

from utils.preprocessor import Preprocessor
from dataset.dataset import Dataset
from models.physiotherapy_model import PhysiotherapyModel
from models.trainer import MultimodalTrainer
from transformers import AutoTokenizer, AutoModel
from torchvision import transforms
import torch
from PIL import Image

from models.transformer import PhysionetForCausalLM, PhysionetConfig, PaliPhysionetConfig

class Physiotherapy:
    def train(self, config_path: dict, utilize_memory: bool = False):

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        torch.autograd.set_detect_anomaly(True)
            
        # Init WandB
        wandb.init(project=config["wandb_project"], config=config)
        # Set up the WandB logger
        wandb_logger = WandbLogger(project=config["wandb_project"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        max_seq_length = config.get("max_seq_length", 2048)
        model_cache_folder = config.get("model_folder", "cache")

        tokenizer_config = config.get("tokenizer", "gpt2")

        # Load the GPT-2 tokenizer using AutoPretrained
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, cache_dir=model_cache_folder)
        self.tokenizer.pad_token = self.tokenizer.eos_token


       # Define the text_config dictionary with descriptive keys
        text_config = {
            "vocab_size": config.get("vocab_size", 257152),
            "hidden_size": config.get("hidden_size", 2048),
            "intermediate_size": config.get("intermediate_size", 2048),
            "num_hidden_layers": config.get("num_hidden_layers", 8),  # Descriptive key
            "num_attention_heads": config.get("num_attention_heads", 8),  # Descriptive key
            "num_key_value_heads": config.get("num_key_value_heads", 8),  # Descriptive key
            "max_position_embeddings": max_seq_length,
            "rms_norm_epsilon": config.get("rms_norm_eps", 1e-6),  # Descriptive key
            "rope_theta": config.get("rope_theta", 10000.0),
            "attention_bias": config.get("attention_bias", False),
            "attention_dropout": config.get("attention_dropout", 0.0),
            "pad_token_id": self.tokenizer.pad_token_id
        }
        

        # Define the vision_config dictionary with descriptive keys
        vision_config = {
            "hidden_size": config.get("vision_hidden_size", 768),
            "intermediate_size": config.get("vision_intermediate_size", 3072),
            "num_of_layers": config.get("vision_num_hidden_layers", 12),  # Descriptive key
            "num_of_attention_heads": config.get("vision_num_attention_heads", 12),  # Descriptive key
            "num_of_channels": config.get("vision_num_channels", 3),  # Descriptive key
            "image_size": config.get("vision_image_size", 224),
            "patch_size": config.get("vision_patch_size", 16),
            "layer_norm_epsilon": config.get("vision_layer_norm_eps", 1e-6),  # Descriptive key
            "attention_dropout": config.get("vision_attention_dropout", 0.0),
        }


        # Load a pretrained Vision Transformer model using AutoModel
        vit_model_name = config.get("vit_model_name", "google/siglip-base-patch16-224")
        self.vit = AutoModel.from_pretrained(vit_model_name, cache_dir=model_cache_folder)

        transformer_config = PaliPhysionetConfig(
            vision_config=vision_config,
            text_config=text_config,
            ignore_index=config.get("ignore_index", -100),
            image_token_index=config.get("image_token_index", 256000),
            vocab_size=config.get("vocab_size", 257152),
            projection_dim=config.get("projection_dim", 2048),
            hidden_size=config.get("hidden_size", 2048),
            pad_token_id=self.tokenizer.pad_token_id,
            num_hidden_layers=config.get("num_hidden_layers", 32),
        )


        # Initialize the GemmaForCausalLM model
        self.model = PhysionetForCausalLM(transformer_config)
        self.model.to(device)
        print(f"Model : {self.model}")

        # Initialize the PhysiotherapyModel with the GemmaForCausalLM model
        physionet = PhysiotherapyModel(
            vit=self.vit,
            max_seq_length=max_seq_length,
            device=device,
            tokenizer=self.tokenizer,
            model=self.model,
            vit_config=vision_config
        )

        multimodal = MultimodalTrainer(
            model=physionet,
            tokenizer=self.tokenizer,
            learning_rate=config["learning_rate"],
            max_epochs=config["max_epochs"]
        )

        trainer = pl.Trainer(
            max_epochs=config["max_epochs"],
            accelerator=config["device"],
            devices=config["gpus"],
            precision=config["precision"],
            gradient_clip_val=1.0,
            accumulate_grad_batches=config["accumulate_grad_batches"],
            log_every_n_steps=config["log_every_n_steps"],
            val_check_interval=config["val_check_interval"],
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=config.get("checkpoints", "checkpoints"),
                    filename="physiotherapy-{epoch:02d}-{val_loss:.2f}",
                    save_top_k=1,
                    every_n_train_steps=20,
                ),
                TQDMProgressBar(
                    refresh_rate=20,
                    process_position=0,
                    leave=True
                ),
                LearningRateMonitor(logging_interval="step")
            ]
        )

        # Check if the model path exists
        model_path = config.get("model_path")  # Add this in your config
        if model_path and os.path.exists(model_path):
            multimodal = MultimodalTrainer.load_from_checkpoint(
                checkpoint_path=model_path,
                model=physionet,
                tokenizer=self.tokenizer
            )

        multimodal.to(device)
        print(f"Combined Model {multimodal}")

        print(f"Set up trainer")
        datamodule = Dataset(
            train_csv_path=config.get("train_csv_path", "dataset/data/train.csv"),
            test_csv_path=config.get("test_csv_path", "dataset/data/test.csv"),
            val_csv_path=config.get("val_csv_path", "dataset/data/val.csv"),
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            resolution=config.get("resolution", 224),
            train_val_split=config.get("train_val_split", 0.8),
            pin_memory=True,
            shuffle=False,
            tokenizer=self.tokenizer,
            model=self.model,
            utilize_memory=utilize_memory,
            file_index=config.get("file_index", 0)
        )

        datamodule = datamodule.get_dataloader()

        print(f"Data module done")
        trainer.fit(model=multimodal, datamodule=datamodule)

        # Setup the datamodule for testing
        datamodule.setup(stage="test")
        trainer.test(model=multimodal, datamodule=datamodule)

        # Finish WandB run
        wandb.finish()

        # Clean up all memory
        torch.cuda.empty_cache()

    def inference(self, config_path: str, video_path: str):
        """
        Perform inference on a video using a trained multimodal model.
        
        Args:
            config_path (str): Path to the configuration YAML file
            video_path (str): Path to the input video file
        
        Returns:
            list: List of model outputs for each frame
        """
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Set up model
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Model path not specified in config")
        
        max_seq_length = config.get("max_seq_length", 2048)
        model_cache_folder = config.get("model_folder", "cache")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        tokenizer_config = config.get("tokenizer", "gpt2")

        # Load the GPT-2 tokenizer using AutoPretrained
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config, cache_dir=model_cache_folder)
        self.tokenizer.pad_token = self.tokenizer.eos_token


        # Load a pretrained Vision Transformer model using AutoModel
        vit_model_name = config.get("vit_model_name", "google/siglip-base-patch16-224")
        self.vit = AutoModel.from_pretrained(vit_model_name, cache_dir=model_cache_folder)


       # Define the text_config dictionary with descriptive keys
        text_config = {
            "vocab_size": config.get("vocab_size", 257152),
            "hidden_size": config.get("hidden_size", 2048),
            "intermediate_size": config.get("intermediate_size", 2048),
            "num_hidden_layers": config.get("num_hidden_layers", 8),  # Descriptive key
            "num_attention_heads": config.get("num_attention_heads", 8),  # Descriptive key
            "num_key_value_heads": config.get("num_key_value_heads", 8),  # Descriptive key
            "max_position_embeddings": max_seq_length,
            "rms_norm_epsilon": config.get("rms_norm_eps", 1e-6),  # Descriptive key
            "rope_theta": config.get("rope_theta", 10000.0),
            "attention_bias": config.get("attention_bias", False),
            "attention_dropout": config.get("attention_dropout", 0.0),
            "pad_token_id": self.tokenizer.pad_token_id
        }
        

        # Define the vision_config dictionary with descriptive keys
        vision_config = {
            "hidden_size": config.get("vision_hidden_size", 768),
            "intermediate_size": config.get("vision_intermediate_size", 3072),
            "num_of_layers": config.get("vision_num_hidden_layers", 12),  # Descriptive key
            "num_of_attention_heads": config.get("vision_num_attention_heads", 12),  # Descriptive key
            "num_of_channels": config.get("vision_num_channels", 3),  # Descriptive key
            "image_size": config.get("vision_image_size", 224),
            "patch_size": config.get("vision_patch_size", 16),
            "layer_norm_epsilon": config.get("vision_layer_norm_eps", 1e-6),  # Descriptive key
            "attention_dropout": config.get("vision_attention_dropout", 0.0),
        }

        transformer_config = PaliPhysionetConfig(
            vision_config=vision_config,
            text_config=text_config,
            ignore_index=config.get("ignore_index", -100),
            image_token_index=config.get("image_token_index", 256000),
            vocab_size=config.get("vocab_size", 257152),
            projection_dim=config.get("projection_dim", 2048),
            hidden_size=config.get("hidden_size", 2048),
            pad_token_id=self.tokenizer.pad_token_id,
            num_hidden_layers=config.get("num_hidden_layers", 32),
        )


        # Initialize the GemmaForCausalLM model
        self.model = PhysionetForCausalLM(transformer_config)
        self.model.to(device)
        print(f"Model : {self.model}")

        physionet = PhysiotherapyModel(
            vit=self.vit,
            max_seq_length=max_seq_length,
            device=device,
            tokenizer=self.tokenizer,
            model=self.model,
            vit_config=vision_config
        )

        physionet.eval()

        model_path = config.get("model_path") 
        if model_path and os.path.exists(model_path):
            multimodal = MultimodalTrainer.load_from_checkpoint(
                checkpoint_path=model_path,
                model=physionet,
                tokenizer=self.tokenizer
            )

        preprocessor = Preprocessor(self.tokenizer, model=self.model)

        multimodal.eval()
        multimodal.to(device)

        # Video setup
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Error: Could not open video file {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing video with {total_frames} frames at {fps} FPS")

        # Configure preprocessing
        resolution = config.get("resolution", 224)
        normalization = config.get("normalisation", ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

        transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])
        ])

        frame_buffer = []
        sequence_outputs = []
        individual_outputs = []
        dummy_frame = np.zeros((3, resolution, resolution), dtype=np.float32)

        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_tensor = transform(frame)
                frame_np = frame_tensor.numpy()
                
                # 1. Process individual frame
                # Create frame array with correct shape [1, n_frames, channels, height, width]
                frame_array = [frame_np]  # First frame is the actual frame
                frame_array.extend([dummy_frame] * 99)  # Add 99 dummy frames
                
                individual_data = {
                    "prompt": [f"{preprocessor.FRAME_TOKEN} {preprocessor.BBX1_TOKEN}22{preprocessor.BBX1_TOKEN}"],
                    "frame": frame_array,  # MultimodalPreprocessor expects list of frames
                    "n_frames": [1],  # Only processing one real frame
                    "size": [resolution],
                    "additional_info": [""],
                    "label": ""
                }

                print(f"Individual frame prompt: {individual_data['prompt']}")
                
                # Process individual frame
                with torch.no_grad():
                    try:
                        individual_output = multimodal.generate(
                            batch=individual_data,
                            max_new_tokens=5,
                            do_sample=False
                        )
                        individual_outputs.append({
                            'frame_idx': frame_count,
                            'output': individual_output
                        })
                        print(f"Individual frame {frame_count} output: {individual_output}")
                    except Exception as e:
                        print(f"Error processing individual frame {frame_count}: {str(e)}")
                
                # 2. Add to sequence buffer for sequence processing
                frame_buffer.append(frame_np)
                
                # Process sequence when we have enough frames or at the end
                if len(frame_buffer) >= 100 or not ret:
                    # Ensure we have exactly 100 frames
                    current_frames = frame_buffer[:100]
                    if len(current_frames) < 100:
                        current_frames.extend([dummy_frame] * (100 - len(current_frames)))
                    
                    frame_prompt = f"{preprocessor.FRAME_TOKEN} "
                    frame_images = [frame_prompt]
                    for i in enumerate(current_frames):
                        frame_images.append(frame_prompt)

                    sequence_data = {
                        "prompt": [f"{''.join(frame_images)}{preprocessor.ACTION_TOKEN}"],
                        "frame": current_frames,  # List of frames for a single batch item
                        "n_frames": [len(current_frames)],  # Number of actual frames
                        "size": [resolution],
                        "additional_info": [""],
                        "label": ""
                    }
                    
                    print(sequence_data["prompt"])

                    # Process sequence
                    with torch.no_grad():
                        try:
                            sequence_output = multimodal.generate(
                                batch=sequence_data,
                                max_new_tokens=10,
                                do_sample=False
                            )
                            sequence_outputs.append({
                                'start_frame': max(0, frame_count - len(frame_buffer)),
                                'end_frame': frame_count,
                                'output': sequence_output
                            })
                            print(f"Sequence output for frames {max(0, frame_count - len(frame_buffer))} to {frame_count}: {sequence_output}")
                        except Exception as e:
                            print(f"Error processing sequence at frame {frame_count}: {str(e)}")
                    
                    # Clear buffer but keep overlap frames
                    frame_buffer = frame_buffer[50:]  # 50% overlap
                
                frame_count += 1

        finally:
            cap.release()

        # Combine outputs into final structured result
        final_result = {
            'individual_frames': individual_outputs,
            'sequences': sequence_outputs,
            'metadata': {
                'total_frames': frame_count,
                'resolution': resolution,
                'video_path': video_path
            }
        }

        return final_result