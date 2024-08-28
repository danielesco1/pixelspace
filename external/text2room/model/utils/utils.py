import torch
import cv2
import os
import json
import numpy as np
import time
import pymeshlab
import imageio

from PIL import Image

from PIL import Image

from diffusers import AutoPipelineForInpainting, UNet2DConditionModel,StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler, StableDiffusionPipeline, StableDiffusionInpaintPipeline
from diffusers import MarigoldDepthPipeline
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

def load_inpainting(args):
    inpaint_model = "sdxl"
    if inpaint_model == "sd":
        model_path = "stabilityai/stable-diffusion-2-inpainting"
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(args.device)

        pipe.set_progress_bar_config(**{
            "leave": False,
            "desc": "Generating Next Image"
        })
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if inpaint_model =="sdxl":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            ).to(args.device)
        pipe.set_progress_bar_config(**{
            "leave": False,
            "desc": "Generating Next Image"
        })
        pipe.enable_model_cpu_offload()
    return pipe
    
def load_depth_model(args):
    depth_model_path = "prs-eth/marigold-depth-lcm-v1-0"
    pipe = MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-0", variant="fp16", torch_dtype=torch.float16
    ).to(args.device)
    pipe.enable_model_cpu_offload()
    return pipe

def generate_first_image(args):
    """
    model_path = os.path.join(args.models_path, "stable-diffusion-2-1")
    
    if not os.path.exists(model_path):
        model_path = "stabilityai/stable-diffusion-2-1"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)
    """
    pipe = DiffusionPipeline.from_pretrained("SG161222/RealVisXL_V4.0", variant="fp16", torch_dtype=torch.float16).to(args.device)
    pipe.enable_model_cpu_offload()
    
    pipe.set_progress_bar_config(**{
        "leave": False,
        "desc": "Generating Start Image"
    })
    
    #generator = torch.manual_seed(-1)
    image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            #generator=generator,
            guidance_scale=8
        ).images[0]
    return image

def load_refiner_model(args):
    pipeline = AutoPipelineForImage2Image.from_pretrained(
            "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
            ).to(args.device)
    pipeline.enable_model_cpu_offload()
    return pipeline

def refine_pipe(image,args):
    # prepare image
    #url = "/home/descobar/projects/xbase/output/text2room/start_room01/full_trajectory/A_room_made_out_of_blue_powdered_walls_w/no_input_image_file/2024-08-21_17:31:30.290947Z/rgbd/rgbd_0002.png"
    #init_image = load_image(url)

    prompt = "A room made out of blue powdered walls with the ceiling open to the blue sky, futuristic furniture and plants"

    # pass prompt and image to pipeline
    image = args.pipeline(prompt, image=image, strength=0.5).images[0]
    #make_image_grid([init_image, image], rows=1, cols=2)

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

def save_settings(args):
    with open(os.path.join(args.out_path, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
        
def pil_to_torch(img, device, normalize=True):
    img = torch.tensor(np.array(img), device=device).permute(2, 0, 1)
    if normalize:
        img = img / 255.0
    return img

def save_image(image, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    image.save(file_out)
    return file_with_ext


def save_rgbd(image, depth, prefix, idx, outdir):
    filename = f"{prefix}_{idx:04}"
    ext = "png"
    file_with_ext = f"{filename}.{ext}"
    file_out = os.path.join(outdir, file_with_ext)
    dst = Image.new('RGB', (image.width + depth.width, image.height))
    dst.paste(image, (0, 0))
    dst.paste(depth, (image.width, 0))
    dst.save(file_out)
    return file_with_ext


def save_settings(args):
    with open(os.path.join(args.out_path, "settings.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)


def save_animation(image_folder_path, prefix=""):
    gif_name = os.path.join(image_folder_path, prefix + 'animation.gif')
    images = [os.path.join(image_folder_path, img) for img in sorted(os.listdir(image_folder_path), key=lambda x: int(x.split(".")[0].split("_")[-1])) if "rgb" in img]

    with imageio.get_writer(gif_name, mode='I', duration=0.2) as writer:
        for filename in images:
            image = imageio.v3.imread(filename)
            writer.append_data(image)


def save_poisson_mesh(mesh_file_path, depth=12, max_faces=10_000_000):
    # load mesh
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file_path)
    print("loaded", mesh_file_path)

    # compute normals
    start = time.time()
    ms.compute_normal_for_point_clouds()
    print("computed normals")

    # run poisson
    ms.generate_surface_reconstruction_screened_poisson(depth=depth)
    end_poisson = time.time()
    print(f"finish poisson in {end_poisson - start} seconds")

    # save output
    parts = mesh_file_path.split(".")
    out_file_path = ".".join(parts[:-1])
    suffix = parts[-1]
    out_file_path_poisson = f"{out_file_path}_poisson_meshlab_depth_{depth}.{suffix}"
    ms.save_current_mesh(out_file_path_poisson)
    print("saved poisson mesh", out_file_path_poisson)

    # quadric edge collapse to max faces
    start_quadric = time.time()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
    end_quadric = time.time()
    print(f"finish quadric decimation in {end_quadric - start_quadric} seconds")

    # save output
    out_file_path_quadric = f"{out_file_path}_poisson_meshlab_depth_{depth}_quadric_{max_faces}.{suffix}"
    ms.save_current_mesh(out_file_path_quadric)
    print("saved quadric decimated mesh", out_file_path_quadric)

    return out_file_path_poisson