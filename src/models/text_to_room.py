import sys
import os
import json
import argparse
from PIL import Image

# Step 1: Add the external repository to the Python path
repo_path = os.path.join(os.path.dirname(__file__), '../../external')
sys.path.append(repo_path)


from text2room.model.utils.utils import generate_first_image, save_poisson_mesh
from text2room.model.text2room_pipeline import Text2RoomPipeline
from utils.utils import save_image, create_output_path
from utils.vis.cameraplots import plot_camera_scene

import torch

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
    args = json_to_args(config)
    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))
    
    pipeline_type = args.pipeline
    #output_path = create_output_path(f"output/{pipeline_type}/{args.output_path}")
    
    #print(output_path)
    
    print(f"trajectories len: {len(trajectories)}")

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        args.prompt = trajectories[0]["prompt"]

    # get first image from text prompt or saved image folder
    if (not args.input_image_path) or (not os.path.isfile(args.input_image_path)):
        first_image_pil = generate_first_image(args)
    else:
        first_image_pil = Image.open(args.input_image_path)
        
    #load pipeline
    pipeline = Text2RoomPipeline(args, first_image_pil=first_image_pil)
    
    offset = 1
    for t in trajectories:
        print(f"traj: {t}")
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)
        poses = pipeline.seen_poses
        cams = pipeline.seen_cameras

        output_path = pipeline.args.output_camera_plots
        cams_path = os.path.join(output_path,f"{offset}traj_{t['fn_name']}_cams.png")
        print(f"saving cams: {cams_path}")
        plot_camera_scene(cams, annotate=True, save_path=cams_path)
        
        #save poses
        pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
        pipeline.save_nerf_transforms()
        print(f"current cam: {pipeline.world_to_cam}")
        
        fused_mesh_path = pipeline.args.fused_mesh_path
        combined_image_path = pipeline.args.combined_image_path
        # Get the latest image and mesh file after each trajectory
        latest_image = sorted(os.listdir(combined_image_path))[-1]
        latest_mesh = sorted(os.listdir(fused_mesh_path))[-1]
        
        latest_image_path = os.path.join(combined_image_path, latest_image)
        latest_mesh_path = os.path.join(fused_mesh_path, latest_mesh)
        
        yield latest_image_path, latest_mesh_path
    
    # save outputs before completion
    pipeline.clean_mesh()
    intermediate_mesh_path = pipeline.save_mesh("after_generation.ply")
    save_poisson_mesh(intermediate_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # run completion
    pipeline.args.update_mask_after_improvement = True
    pipeline.complete_mesh(offset=offset)
    pipeline.clean_mesh()

    # Now no longer need the models
    pipeline.remove_models()

    # save outputs after completion
    final_mesh_path = pipeline.save_mesh()
    
    """
    # run poisson mesh reconstruction
    mesh_poisson_path = save_poisson_mesh(final_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)
    """
    
    mesh_poisson_path = final_mesh_path
    # save additional output
    pipeline.save_animations()
    pipeline.load_mesh(mesh_poisson_path)
    pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
    pipeline.save_nerf_transforms()
    pipeline.save_seen_trajectory_renderings(apply_noise=True)

    print("Finished. Outputs stored in:", args.out_path)