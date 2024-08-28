import argparse
import json

from utils.utils import save_image, create_output_path

import sys
import os

import numpy as np
import torch
from PIL import Image  # Import PIL for image handling
# Add the external repository to the Python path
# Step 1: Add the external repository to the Python path

repo_path = os.path.join(os.path.dirname(__file__), '../external')
sys.path.append(repo_path)

from models import (
    image_to_3d, 
    text_to_room, 
    invisible_stitch, 
    eschernet)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def run(config):
    pipeline_type = config['pipeline']
    output_path = create_output_path(f"output/{config['output_path']}")

    pipelines = {
        #"text_to_image": text_to_image.run_pipeline,
        #"controlnet": controlnet.run_pipeline,
        #"inpaint": inpaint.run_pipeline,
        "image_to_3d": image_to_3d.run_pipeline,
        "text_to_room": text_to_room.generate_scene,
        "invisible_stitch": invisible_stitch.generate_scene,
        "eschernet": eschernet.generate_scene
    }
    
    pipeline_function = pipelines.get(pipeline_type)
    if not pipeline_function:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    output = pipeline_function(config)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different diffusion pipelines.")
    parser.add_argument('--config', required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    config = load_config(f"src/config/{args.config}")
    
    pipeline_type = config['pipeline']

    run(config)
    