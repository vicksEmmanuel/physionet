import gradio as gr
import os
from physionet_paligemma import PhysiotherapyPaligemmaConfig
from physionet import Physiotherapy
import uuid

physiotherapy = PhysiotherapyPaligemmaConfig(
    "config/physionet.yaml"
)

def run_inference(video_path):
    """
    Function to run inference on the input video.
    """
    output_video_path = f"outputs/{str(uuid.uuid4())}.mp4"

    # Call your inference function
    inference_result = physiotherapy.inference(
        "config/physionet.yaml",  # Path to your config file
        video_path,               # Input video path
        output_video_path         # Output video path
    )

    # Assuming `inference_result` contains both the output video path and some text data
    # For example: inference_result = {"video_path": "output.mp4", "text": "Detected actions: ..."}
    return inference_result

# Gradio interface
def gradio_interface(video):
    """
    Gradio interface function to handle video input and output.
    """
    # Run inference on the input video
    inference_result = run_inference(video)

    # Extract the output video path and text from the inference result
    output_video_path = inference_result["video_path"]
    output_text = inference_result["text"]

    # Return the output video file and text
    return output_video_path, output_text

# Create the Gradio interface
interface = gr.Interface(
    fn=gradio_interface,  # Function to handle the input and output
    inputs=gr.Video(label="Upload Video"),  # Input: Video file
    outputs=[
        gr.Video(label="Processed Video"),  # Output: Processed video file
        gr.Textbox(label="Inference Results")  # Output: Text output
    ],
    title="Physionet",
    description="Upload a video for physiotherapy action detection and processing."
)

# Launch the Gradio interface
interface.launch()