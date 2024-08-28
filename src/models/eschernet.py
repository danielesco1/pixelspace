import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import skimage
from PIL import Image

import gradio as gr
import json
import argparse

from torchvision import transforms
from tqdm.auto import tqdm
import torchvision

# Step 1: Add the external repository to the Python path
repo_path = os.path.join(os.path.dirname(__file__), '../../external')
sys.path.append(repo_path)

def json_to_args(json_data):
    # Create an argparse namespace object
    args = argparse.Namespace()
    
    # Convert top-level keys
    for key, value in json_data.items():
        if isinstance(value, dict):
            # If the value is a dict, we assume it's a group of arguments
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            # Otherwise, it's a top-level argument
            setattr(args, key, value)
    
    return args

def generate_scene(config):
    current_directory = os.getcwd()
    print(f"Running script in directory: {current_directory}")
    args = json_to_args(config)

    pipeline_type = args.pipeline
    CaPE_TYPE = args.mode
    print(pipeline_type, args.mode)
    weight_dtype = torch.float16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sys.path.insert(0, "./external/eschernet/sixDoF/")
    
    #print("Python search paths:", sys.path)

    # use the customized diffusers modules
    from diffusers import DDIMScheduler
    from dataset import get_pose
    from CN_encoder import CN_encoder
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline
    
    # loading pipelines
    print(f"loading {args.pretrained_model_name_or_path}")
    # Init pipeline
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler",
                                              revision=args.revision)
    image_encoder = CN_encoder.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder", revision=args.revision)
    
    print(f"loading model...")
    pipeline = Zero1to3StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        scheduler=scheduler,
        image_encoder=None,
        safety_checker=None,
        feature_extractor=None,
        torch_dtype=weight_dtype,
    )
    pipeline.image_encoder = image_encoder
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    #print(f"encoder: {image_encoder}")
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    DATA_DIR = args.data_dir
    DATA_TYPE = args.data_type
    
    print(DATA_DIR)

    if DATA_TYPE == "GSO25":
        T_in_DATA_TYPE = "render_mvs_25" # same condition for GSO
        T_out_DATA_TYPE = "render_mvs_25"   # for 2D metrics
        T_out = 25
        
    T_in = args.T_in
    OUTPUT_DIR= f"{args.output_dir}logs_{CaPE_TYPE}/{DATA_TYPE}/N{T_in}M{T_out}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Init Dataset
    image_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.resolution, args.resolution)),  # 256, 256
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
    )
    #get object dataset
    obj_names = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
    print(obj_names)
    
    

    