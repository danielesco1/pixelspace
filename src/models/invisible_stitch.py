import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import skimage
from PIL import Image

import gradio as gr

# Step 1: Add the external repository to the Python path
repo_path = os.path.join(os.path.dirname(__file__), '../../external')
sys.path.append(repo_path)

from invisible_stitch.utils.render import PointsRendererWithMasks, render
from invisible_stitch.utils.ops import snap_high_gradients_to_nn, project_points, get_pointcloud, merge_pointclouds, outpaint_with_depth_estimation

from invisible_stitch.utils.ops import focal2fov, fov2focal, convert_depth_to_image
from invisible_stitch.utils.models import infer_with_zoe_dc,get_sd_pipeline, get_zoe_dc_model
from invisible_stitch.utils.gs import gs_options, read_cameras_from_optimization_bundle, Scene, run_gaussian_splatting, get_blank_gs_bundle
from invisible_stitch.utils.scene.gaussian_model import GaussianModel
#from invisible_stitch.utils.demo import downsample_point_cloud

from typing import Iterable, Tuple, Dict, Optional, Literal
import itertools

from tqdm.auto import tqdm
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
)

import json
import argparse

from utils.utils import save_image, create_output_path
from utils.vis.cameraplots import plot_camera_scene
from diffusers.utils import make_image_grid, load_image
import cv2

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
"""
def setup_output_directories():
        prompt_folder_name = self.orig_prompt[:40]
        prompt_folder_name = prompt_folder_name.replace(" ", "_")

        if os.path.isfile(self.args.trajectory_file):
            trajectory_name = os.path.basename(self.args.trajectory_file)
            trajectory_name = trajectory_name.split(".")[0]
            self.args.out_path = os.path.join(self.args.out_path, trajectory_name)

        self.args.out_path = os.path.join(self.args.out_path, self.args.exp_name, prompt_folder_name)

        if self.args.input_image_path:
            file_name = self.args.input_image_path.split("/")[-1]
            self.args.out_path = os.path.join(self.args.out_path, file_name)
        else:
            self.args.out_path = os.path.join(self.args.out_path, "no_input_image_file")

        now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%fZ')
        self.args.out_path = os.path.join(self.args.out_path, now_str)
        os.makedirs(self.args.out_path, exist_ok=True)
        self.args.rgb_path = os.path.join(self.args.out_path, "rgb")
        self.args.rgbd_path = os.path.join(self.args.out_path, "rgbd")
        self.args.rendered_path = os.path.join(self.args.out_path, "rendered")
        self.args.depth_path = os.path.join(self.args.out_path, "depth")
        self.args.fused_mesh_path = os.path.join(self.args.out_path, "fused_mesh")
        self.args.mask_path = os.path.join(self.args.out_path, "mask")
        self.args.output_rendering_path = os.path.join(self.args.out_path, "output_rendering")
        self.args.output_depth_path = os.path.join(self.args.out_path, "output_depth")
        os.makedirs(self.args.rgb_path, exist_ok=True)
        os.makedirs(self.args.rgbd_path, exist_ok=True)
        os.makedirs(self.args.rendered_path, exist_ok=True)
        os.makedirs(self.args.depth_path, exist_ok=True)
        os.makedirs(self.args.fused_mesh_path, exist_ok=True)
        os.makedirs(self.args.mask_path, exist_ok=True)
        os.makedirs(self.args.output_rendering_path, exist_ok=True)
        os.makedirs(self.args.output_depth_path, exist_ok=True)
        
        self.args.output_camera_plots = os.path.join(self.args.out_path, "output_camera_plots")
        self.args.combined_image_path = os.path.join(self.args.out_path, "output_combined_img")
        os.makedirs(self.args.output_camera_plots, exist_ok=True)
        os.makedirs(self.args.combined_image_path, exist_ok=True)
        save_settings(self.args)
"""

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_TURBO):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        if (x > 0).any():
            mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
            ma = np.max(x)
        else:
            mi = 0.0
            ma = 0.0
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi, ma]

def extrapolate_point_cloud(prompt: str, image_size: Tuple[int, int], look_at_params: Iterable[Tuple[float, float, float, Tuple[float, float, float]]], point_cloud: Pointclouds = None, dry_run: bool = False, discard_mask: bool = False, initial_image: Optional[Image.Image] = None, depth_scaling: float = 1, seed: int = 0, **render_kwargs):
    w, h = image_size
    optimization_bundle_frames = []
    track_frames = 1
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    cams = []
    for azim, elev, dist, at in tqdm(look_at_params):
        R, T = look_at_view_transform(device=device, azim=azim, elev=elev, dist=dist, at=at)
        cameras = PerspectiveCameras(R=R, T=T, focal_length=torch.tensor([w], dtype=torch.float32), principal_point=(((h-1)/2, (w-1)/2),), image_size=(image_size,), device=device, in_ndc=False)
        cams.append(cameras)
        if point_cloud is not None:
            images, masks, depths = render(cameras, point_cloud, **render_kwargs)

            if not dry_run:
                eroded_mask = skimage.morphology.binary_erosion((depths[0] > 0).cpu().numpy(), footprint=None)#skimage.morphology.disk(1))
                eroded_depth = depths[0].clone()
                eroded_depth[torch.from_numpy(eroded_mask).to(depths.device) <= 0] = 0

                inpaint_mask_pil = Image.fromarray(masks[0].detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
                #depth_pil = Image.fromarray(visualize_depth_numpy(eroded_depth[0].cpu().numpy())[0].astype(np.uint8))
                image_array = (images[0].detach().cpu().squeeze().float().numpy() * 255).astype(np.uint8)
                #print(image_array.shape, eroded_depth.shape)
                image_pil = Image.fromarray(image_array).convert("RGB")
                
               
                # Use the function to convert out_depth to a PIL image
                depth_pil_image = convert_depth_to_image(eroded_depth[0].cpu().numpy())
                #combined_img = make_image_grid([image_pil, inpaint_mask_pil, depth_pil_image], rows=1, cols=3)
                
                #save_image(combined_img, f'{track_frames}_before_{prompt}', seed, output_path)
                
                print(f"outpaint_with_depth_estimation....")
                outpainted_img, aligned_depth = outpaint_with_depth_estimation(images[0], masks[0], 
                                                                               eroded_depth, h, w, 
                                                                               pipe, 
                                                                               zoe_dc_model, 
                                                                               prompt, 
                                                                               cameras, 
                                                                               dilation_size=2, 
                                                                               depth_scaling=depth_scaling, 
                                                                               generator=generator, 
                                                                               output_path=output_path,
                                                                               track_frames=track_frames)

                aligned_depth = torch.from_numpy(aligned_depth).to(device)

            else:
                # in a dry run, we do not actually outpaint the image
                outpainted_img = Image.fromarray((255*images[0].cpu().numpy()).astype(np.uint8))

        else:
            assert initial_image is not None
            assert not dry_run
            print("jumpstart the point cloud with a regular depth estimation")
            # jumpstart the point cloud with a regular depth estimation
            t_initial_image = torch.from_numpy(np.asarray(initial_image)/255.).permute(2,0,1).float()
            depth = aligned_depth = infer_with_zoe_dc(zoe_dc_model, t_initial_image, torch.zeros(h, w))
            outpainted_img = initial_image
            images = [t_initial_image.to(device)]
            masks = [torch.ones(h, w, dtype=torch.bool).to(device)]
     
            inpaint_mask_pil = Image.fromarray(masks[0].detach().cpu().squeeze().float().numpy() * 255).convert("RGB")
            depth_pil = Image.fromarray(visualize_depth_numpy(aligned_depth[0].cpu().numpy())[0].astype(np.uint8))
            
            #combined_img = make_image_grid([outpainted_img, depth_pil], rows=1, cols=2)
            
            #save_image(combined_img, f'{track_frames}_after_{prompt}', seed, output_path)
        
        
        if not dry_run:
            print(f"snap_high_gradients_to_nn....")
            # snap high gradients to nearest neighbor, which eliminates noodle artifacts
            aligned_depth = snap_high_gradients_to_nn(aligned_depth.to(device), threshold=12).cpu()
            xy_depth_world = project_points(cameras, aligned_depth)

        c2w = cameras.get_world_to_view_transform().get_matrix()[0]

        optimization_bundle_frames.append({
            "image": outpainted_img,
            "mask": masks[0].cpu().numpy(),
            "transform_matrix": c2w.tolist(),
            "azim": azim,
            "elev": elev,
            "dist": dist,
        })

        if discard_mask:
            optimization_bundle_frames[-1].pop("mask")

        if not dry_run:
            optimization_bundle_frames[-1]["center_point"] = xy_depth_world[0].mean(dim=0).tolist()
            optimization_bundle_frames[-1]["depth"] = aligned_depth.cpu().numpy()
            optimization_bundle_frames[-1]["mean_depth"] = aligned_depth.mean().item()

        else:
            # in a dry run, we do not modify the point cloud
            continue

        rgb = (torch.from_numpy(np.asarray(outpainted_img).copy()).reshape(-1, 3).float() / 255).to(device)

        if point_cloud is None:
            point_cloud = get_pointcloud(xy_depth_world[0], device=device, features=rgb)

        else:
            # pytorch 3d's mask might be slightly too big (subpixels), so we erode it a little to avoid seams
            # in theory, 1 pixel is sufficient but we use 2 to be safe
            masks[0] = torch.from_numpy(skimage.morphology.binary_erosion(masks[0].cpu().numpy(), footprint=skimage.morphology.disk(2))).to(device)

            partial_outpainted_point_cloud = get_pointcloud(xy_depth_world[0][~masks[0].view(-1)], device=device, features=rgb[~masks[0].view(-1)])

            point_cloud = merge_pointclouds([point_cloud, partial_outpainted_point_cloud])

        track_frames+=1
        
        # Save plot every 2 iterations
        if track_frames % 2 == 0:
            cams_path = os.path.join(output_path, f"cams_{track_frames}__{prompt[:30]}.png")
            print(f"saving cams: {cams_path}")
            plot_camera_scene(cams, annotate=True, save_path=cams_path)
    
    return optimization_bundle_frames, point_cloud

def generate_point_cloud(initial_image: Image.Image, prompt: str, mode: Literal["single", "stage", "360"] = "stage", seed: int = 0):
    image_size = initial_image.size
    w, h = image_size

    optimization_bundle = get_blank_gs_bundle(h, w)

    step_size = 25

    if mode == "single":
        azim_steps = [0]
    elif mode == "stage":
        azim_steps = [0, step_size, -step_size]
    elif mode == "360":
        azim_steps = [x for x in range(0, 360, step_size) if x < 272.5] + [272.5, 316.25]

    look_at_params = [(azim, 0, 0.01, torch.zeros((1, 3))) for azim in azim_steps]

    optimization_bundle["frames"], point_cloud = extrapolate_point_cloud(prompt, image_size, look_at_params, discard_mask=True, initial_image=initial_image, depth_scaling=0.5, seed=seed, fill_point_cloud_holes=True)

    optimization_bundle["pcd_points"] = point_cloud.points_padded()[0].cpu().numpy()
    optimization_bundle["pcd_colors"] = point_cloud.features_padded()[0].cpu().numpy()

    return optimization_bundle, point_cloud

def supplement_point_cloud(optimization_bundle: Dict, point_cloud: Pointclouds, prompt: str):
    w, h = optimization_bundle["W"], optimization_bundle["H"]

    supporting_frames = []

    for i, frame in enumerate(tqdm(optimization_bundle["frames"])):
        # skip supporting views
        if frame.get("supporting", False):
            continue

        center_point = torch.tensor(frame["center_point"]).to(device)
        mean_depth = frame["mean_depth"]
        azim, elev = frame["azim"], frame["elev"]

        azim_jitters = torch.linspace(-5, 5, 3).tolist()
        elev_jitters = torch.linspace(-5, 5, 3).tolist()

        # build the product of azim and elev jitters
        camera_jitters = [{"azim": azim + azim_jitter, "elev": elev + elev_jitter} for azim_jitter, elev_jitter in itertools.product(azim_jitters, elev_jitters)]

        look_at_params = [(camera_jitter["azim"], camera_jitter["elev"], mean_depth, center_point.unsqueeze(0)) for camera_jitter in camera_jitters]

        local_supporting_frames, point_cloud = extrapolate_point_cloud(prompt, (w, h), look_at_params, point_cloud, dry_run=True, depth_scaling=0.5, antialiasing=3)

        for local_supporting_frame in local_supporting_frames:
            local_supporting_frame["supporting"] = True

        supporting_frames.extend(local_supporting_frames)

    optimization_bundle["pcd_points"] = point_cloud.points_padded()[0].cpu().numpy()
    optimization_bundle["pcd_colors"] = point_cloud.features_padded()[0].cpu().numpy()

    return optimization_bundle, point_cloud

def generate_scene(config):
    
    args = json_to_args(config)

    pipeline_type = args.pipeline
    
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    
    global pipe
    pipe = get_sd_pipeline(device)
    
    global zoe_dc_model
    from huggingface_hub import hf_hub_download
    zoe_dc_model = get_zoe_dc_model(ckpt_path=hf_hub_download(repo_id="paulengstler/invisible-stitch", filename="invisible-stitch.pt")).to(device)

    from datetime import datetime
    now_str = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%fZ')
    global output_path
    output_path = create_output_path(os.path.join(args.output_path, now_str))
    
    orig_img = Image.open(args.image).convert("RGB")
    orig_img = orig_img.resize((512,512))

    # crop to ensure the image dimensions are divisible by 8
    img = orig_img.crop((0, 0, orig_img.width - orig_img.width % 8,orig_img.height - orig_img.height % 8))
    
    print(args.prompt, args.mode, args.seed)
    
    pbar = tqdm(total=3)
    pbar.set_description("Hallucinating Scene")

    gs_optimization_bundle, point_cloud = generate_point_cloud(img, args.prompt, mode=str(args.mode), seed=args.seed)
    
    pbar.update(1)
    pbar.set_description("Generating Additional Views")

    #supp_gs_optimization_bundle, _ = supplement_point_cloud(gs_optimization_bundle, point_cloud, args.prompt)
    
    pbar.update(1)
    pbar.set_description("Gaussian Splat Optimization")

    #scene = Scene(supp_gs_optimization_bundle, GaussianModel(gs_options.sh_degree), gs_options)
    scene = Scene(gs_optimization_bundle, GaussianModel(gs_options.sh_degree), gs_options)

    scene = run_gaussian_splatting(scene, gs_optimization_bundle)

    pbar.update(1)

    # coordinate system transformation
    scene.gaussians._xyz = scene.gaussians._xyz.detach()
    scene.gaussians._xyz[:, 1] = -scene.gaussians._xyz[:, 1]
    scene.gaussians._xyz[:, 2] = -scene.gaussians._xyz[:, 2]

    gaussian_path = os.path.join(output_path,f"splat_{args.mode}.ply")
    scene.gaussians.save_ply(gaussian_path)
    
