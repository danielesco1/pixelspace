import argparse
import json

from utils.utils import save_image, create_output_path
from utils.vis.cameraplots import plot_camera_scene
from utils.vis.meshplots import visualize_mesh
from utils.convert_to_nerf_convention import convert_pose_to_nerf_convention
import tempfile
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras
)

import sys
import os

import numpy as np
import rembg
import torch
from PIL import Image  # Import PIL for image handling
# Add the external repository to the Python path
# Step 1: Add the external repository to the Python path
repo_path = os.path.join(os.path.dirname(__file__), '../../external')
sys.path.append(repo_path)

import tempfile

# Step 2: Import the external modules
from sf3d.system import SF3D
import sf3d.utils as sf3d_utils


def load_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)
    
def create_batch(input_image: Image, c2w_cond, intrinsic, intrinsic_normed_cond, COND_WIDTH, COND_HEIGHT, BACKGROUND_COLOR) -> dict:
    img_cond = (
        torch.from_numpy(
            np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32)
            / 255.0
        )
        .float()
        .clip(0, 1)
    )
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(
        torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
    )

    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    # Add batch dim
    batched = {k: v.unsqueeze(0) for k, v in batch_elem.items()}
    return batched

def show_mask_img(input_image: Image) -> Image:
    img_numpy = np.array(input_image)
    alpha = img_numpy[:, :, 3] / 255.0
    chkb = checkerboard(32, 512) * 255
    new_img = img_numpy[..., :3] * alpha[:, :, None] + chkb * (1 - alpha[:, :, None])
    return Image.fromarray(new_img.astype(np.uint8), mode="RGB")
    
def run_pipeline(config):
    pipeline_type = config['pipeline']
    output_path = create_output_path(f"output/{config['output_path']}")
    
    # Get the file name without the suffix
    filename = os.path.splitext(os.path.basename(config["image"]))[0]

    rembg_session = rembg.new_session()
    print(f"testing")
    
    COND_WIDTH = 512
    COND_HEIGHT = 512
    COND_DISTANCE = 1.6
    COND_FOVY_DEG = 40
    BACKGROUND_COLOR = [0.5, 0.5, 0.5]
    
    texture_size = 1024
    remesh_option = "None"
    
    foreground_ratio = 0.85

    # Cached. Doesn't change
    c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
    intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
        COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
    )

    #print(f"c2w_cond: {c2w_cond.shape} {c2w_cond}")
    
    #load SF3D model
    model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
    )
    model.eval()
    model = model.cuda()
    #print(model)
    
    # Load the image using PIL
    image_path = config["image"]  # Replace with the actual path to your image
    
    image = sf3d_utils.remove_background(
                Image.open(image_path).convert("RGBA"), rembg_session
            )
    input_image = sf3d_utils.resize_foreground(image, foreground_ratio)
    
    input_image.save(os.path.join(output_path, f"bg_{filename}_input.png"))
    
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            model_batch = create_batch(input_image, c2w_cond, intrinsic, intrinsic_normed_cond, COND_WIDTH, COND_HEIGHT, BACKGROUND_COLOR)
            model_batch = {k: v.cuda() for k, v in model_batch.items()}
            trimesh_mesh, _glob_dict = model.generate_mesh(
                model_batch, texture_size, remesh_option
            )
            trimesh_mesh = trimesh_mesh[0]

    #print(trimesh_mesh)
       
    output_file = os.path.join(output_path, f"{filename}.glb")

    # Export the mesh directly to the output path
    trimesh_mesh.export(output_file, file_type="glb", include_normals=True)
    
    # Wrap the c2w_cond matrix in a PerspectiveCameras object
    # Reshape c2w_cond to have a batch dimension
    c2w_cond_batched = convert_pose_to_nerf_convention(c2w_cond)
    c2w_cond_batched = c2w_cond.unsqueeze(0)
    cameras = PerspectiveCameras(R=c2w_cond_batched[:,:3, :3], T=c2w_cond_batched[:,:3, 3], device="cuda")
    
    # Visualize the camera pose
    #plot_camera_scene(cameras, "Camera Visualization")
    #visualize_mesh(output_file)
    
    return output_file
    