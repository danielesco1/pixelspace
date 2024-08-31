
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import skimage
from PIL import Image

import gradio as gr
import sys
import os

import argparse
import json
import tempfile

from utils.utils import save_image, create_output_path
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
    return output
    

if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser(description="Run different diffusion pipelines.")
    parser.add_argument('--config', required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    config = load_config(f"src/config/{args.config}")
    
    pipeline_type = config['pipeline']
    """
    #run(config)
    
    import gradio as gr

    def process_image():
        return None
    # Function to update the configuration based on user input and run the pipeline
    def process_image_to_3d(image):
        # Load the default configuration

        config = load_config(f"src/config/configsf3d.json") 
        
        # Save the image to a temporary file and get the path
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            image.save(tmpfile.name)
            image_path = tmpfile.name

        # Update the configuration dictionary with the provided image path
        config["image"] = image_path
        
        # Run the pipeline using the image_to_3d module's run_pipeline method
        output = image_to_3d.run_pipeline(config)
        
        # Clean up the temporary file if desired
        os.remove(image_path)
        return output
    
    
    # List of function names, their corresponding required inputs, and images
    function_inputs = {
        "forward": {"inputs": ["height", "rot", "txmax"], "image": "src/notebooks/camera_plots/forward.png"},
        "forward_small": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/forward_small.png"},
        "left_right": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/left_right.png"},
        "backward": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/backward.png"},
        "backward2": {"inputs": ["height", "rot", "txmax"], "image": "src/notebooks/camera_plots/backward2.png"},
        "backward2_small": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/backward2_small.png"},
        "backward3": {"inputs": ["height", "rot", "txmax"], "image": "src/notebooks/camera_plots/backward3.png"},
        "backward3_small": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/backward3_small.png"},
        "rot_left_up_down": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/rot_left_up_down.png"},
        "back_and_forth_forward": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/back_and_forth_forward.png"},
        "back_and_forth_backward": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/back_and_forth_backward.png"},
        "back_and_forth_forward_reverse": {"inputs": ["height", "rot", "tzmax"], "image": "src/notebooks/camera_plots/back_and_forth_forward_reverse.png"},
        "back_and_forth_forward_reverse_small": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/back_and_forth_forward_reverse_small.png"},
        "back_and_forth_backward_reverse": {"inputs": ["height", "rot", "tzmax"], "image": "src/notebooks/camera_plots/back_and_forth_backward_reverse.png"},
        "back_and_forth_backward_reverse_small": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/back_and_forth_backward_reverse_small.png"},
        "back_and_forth_backward_reverse2": {"inputs": ["height", "rot"], "image": "src/notebooks/camera_plots/back_and_forth_backward_reverse2.png"},
        "sphere_rot": {"inputs": ["radius", "height", "phi"], "image": "src/notebooks/camera_plots/sphere_rot.png"},
        "double_rot": {"inputs": ["radius", "height", "phi"], "image": "src/notebooks/camera_plots/double_rot.png"}
    }

    json_entries = []

    # Function to dynamically show the appropriate inputs and update image
    def update_inputs(fn_name):
        # Initialize all inputs as not visible
        height_visible = gr.update(visible=False)
        rot_visible = gr.update(visible=False)
        txmax_visible = gr.update(visible=False)
        radius_visible = gr.update(visible=False)
        phi_visible = gr.update(visible=False)
        tzmax_visible = gr.update(visible=False)
        
        # Set the relevant inputs to visible based on the selected function
        required_inputs = function_inputs.get(fn_name, {}).get("inputs", [])
        if "height" in required_inputs:
            height_visible = gr.update(visible=True)
        if "rot" in required_inputs:
            rot_visible = gr.update(visible=True)
        if "txmax" in required_inputs:
            txmax_visible = gr.update(visible=True)
        if "radius" in required_inputs:
            radius_visible = gr.update(visible=True)
        if "phi" in required_inputs:
            phi_visible = gr.update(visible=True)
        if "tzmax" in required_inputs:
            tzmax_visible = gr.update(visible=True)
        
        # Update the function image
        function_image = function_inputs.get(fn_name, {}).get("image", None)
        
        return height_visible, rot_visible, txmax_visible, radius_visible, phi_visible, tzmax_visible, function_image

    # Function to add more entries dynamically and update JSON display
    def add_entry(prompt, fn_name, height, rot, txmax, radius, phi, tzmax, adaptive):
        entry = {
            "prompt": prompt,
            "fn_name": fn_name,
            "fn_args": {}
        }
        required_inputs = function_inputs[fn_name]["inputs"]
        if "height" in required_inputs:
            entry["fn_args"]["height"] = height
        if "rot" in required_inputs:
            entry["fn_args"]["rot"] = rot
        if "txmax" in required_inputs:
            entry["fn_args"]["txmax"] = txmax
        if "radius" in required_inputs:
            entry["fn_args"]["radius"] = radius
        if "phi" in required_inputs:
            entry["fn_args"]["phi"] = phi
        if "tzmax" in required_inputs:
            entry["fn_args"]["tzmax"] = tzmax
        if adaptive:
            entry["adaptive"] = [{"arg": "radius", "delta": 0.3}]
        
        json_entries.append(entry)
        return json.dumps(json_entries, indent=2)
    
    # Function to generate JSON and save it as a temporary file
    def generate_json():
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
        json.dump(json_entries, temp_file, indent=2)
        temp_file.close()
        return temp_file.name
    
    # Function to load JSON file and display it
    def load_json(file):
        if file is None:
            return ""
        with open(file.name, 'r') as f:
            loaded_json = json.load(f)
        json_entries.extend(loaded_json)  # Extend the existing entries with the loaded JSON
        return json.dumps(loaded_json, indent=2)
    
    # Function to download the JSON file
    def download_json():
        temp_file_path = generate_json()
        with open(temp_file_path, 'rb') as f:
            return f.read(), temp_file_path
    
    # Global list to hold JSON entries
    json_entries = []

    def clear_everything_fn():
        global mesh_files, image_files
        mesh_files.clear()
        image_files.clear()
        return "", "forward", None, 0, 0, 0, 0, 0, 0, False, "", [], [], None, None

    
    def convert_ply_to_glb(ply_path):
        import trimesh
        # Load the PLY file using trimesh
        mesh = trimesh.load(ply_path)
        
        # Create a temporary file to save the GLB
        glb_path = tempfile.NamedTemporaryFile(delete=False, suffix=".glb").name
        
        # Export the mesh to GLB format
        mesh.export(glb_path)
        
        return glb_path
    
    # Global variable for cancellation flag
    cancel_flag = False

    # Directory containing example JSON files
    EXAMPLES_DIR = "external/text2room/model/trajectories/examples"

    # Function to list example JSON files
    def list_json_files(directory):
        return [f for f in os.listdir(directory) if f.endswith('.json')]

    # Function to load selected example JSON file
    def load_example_json(selected_file):
        file_path = os.path.join(EXAMPLES_DIR, selected_file)
        with open(file_path, 'r') as file:
            json_content = file.read()
        return json_content

    # Initialize the cancellation flag
    cancel_flag = False
    import trimesh 
    def convert_ply_to_glb(ply_path):
        # Load the PLY file using trimesh
        mesh = trimesh.load(ply_path)
        
        # Create a temporary file to save the GLB
        glb_path = tempfile.NamedTemporaryFile(delete=False, suffix=".glb").name
        
        # Export the mesh to GLB format
        mesh.export(glb_path)
        
        return glb_path

    def process_text2room(json_data, json_file_name, n_images, save_scene_every_nth):
        global cancel_flag

        # Load the default configuration
        config = load_config("src/config/configt2room.json")

        # Update the configuration with the new inputs
        config["general"]["n_images"] = n_images
        config["general"]["save_scene_every_nth"] = save_scene_every_nth

        # Extract the output path from the configuration
        out_path = config["general"]["out_path"]

        # Create a new folder for JSON files under the output path
        json_output_path = os.path.join(out_path, "jsons")
        os.makedirs(json_output_path, exist_ok=True)

        # Save the JSON data to a file with the same name as the input JSON file name
        json_file_path = os.path.join(json_output_path, f"{json_file_name}.json")
        with open(json_file_path, 'w') as json_file:
            json.dump(json.loads(json_data), json_file, indent=2)

        # Update the trajectory file path in the configuration
        config["general"]["trajectory_file"] = json_file_path

        # Parse the JSON data to count the total number of trajectories
        trajectory_data = json.loads(json_data)
        total_trajectories = len(trajectory_data)

        # Variable to track if initial mesh files were already loaded
        initial_mesh_loaded = False
        completed_trajectories = 0

        # Run the pipeline and get progressive updates
        for latest_image_path, latest_mesh_path in text_to_room.generate_scene(config):
            if cancel_flag:
                cancel_flag = False  # Reset the flag
                return None, None, f"Cancelled after {completed_trajectories}/{total_trajectories} trajectories"  # Early exit

            # Convert the latest PLY file to GLB
            latest_mesh_glb = convert_ply_to_glb(latest_mesh_path)

            # Increment the completed trajectories count
            completed_trajectories += 1

            # Update the status message
            status_message = f"Completed {completed_trajectories}/{total_trajectories} trajectories"

            # Yield the image, the latest mesh, and the status message
            yield latest_image_path, latest_mesh_glb, status_message

    # Function to cancel the current run
    def cancel_run():
        global cancel_flag
        cancel_flag = True
        return "Process canceled."

    # Function to clear all inputs and outputs
    def clear_everything_fn():
        return "", "forward", None, 0, 0, 0, 0, 0, 0, False, "", None, None

    # Function to download the JSON file with a specified name
    def download_json_with_name(json_data, file_name):
        # Create a temporary file with the specified name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", prefix=file_name)
        # Dump the JSON data into the file
        json.dump(json.loads(json_data), temp_file, indent=2)
        # Close the file to save the changes
        temp_file.close()
        # Return the path of the temporary file for download
        return temp_file.name

    
    # Function to update the JSON display
    def update_json_display():
        return json.dumps(json_entries, indent=2)

    with gr.Blocks() as demo:
        with gr.Tabs():
            
            """
            # Original Interface for generating a scene
            with gr.Tab("SF3d"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(label="Input Image", sources=["upload", "clipboard"], type="pil")
                        
                        submit_button = gr.Button("Generate 3D Scene")
                        examples=[
                        ["samples/images/raccoon_wizard.png", "a suburban street in North Carolina on a bright, sunny day"],
                    ]
                        
                    with gr.Column():
                        output_3d = gr.Model3D(
                            camera_position=(0, 90.0, 1.25),
                            elem_classes="viewport",
                            label="Generated 3D Scene",
                            interactive=False,
                            clear_color=[0.0, 0.0, 0.0, 0.0],
                            scale=1.0,
                        )

                # Connect the submit button to the function
                submit_button.click(
                    fn=process_image_to_3d,
                    inputs=[image_input],
                    outputs=output_3d  # You might need to add outputs=output_files if the function returns multiple outputs
                )
                # Add examples at the bottom of the page
                gr.Examples(
                    examples=[
                        ["samples/images/raccoon_wizard.png"],
                        ["samples/images/u1118458334_a_frontal_elevation_of_a_conceptual_two_story_luis__d4e94706-0710-49e4-bca1-af476ceb1acd.png"]
                    ],
                    inputs=[image_input],
                    outputs=[output_3d],
                    label="Examples of Input Images",
                )
            """
            
            with gr.Tab("pixelspace"):
                # Scene generation interface
                with gr.Column():
                    output_3d = gr.Model3D(label="Generated Scene",
                                           camera_position=(0, 90.0, 1.25),
                                           elem_classes="viewport",
                                           interactive=False,
                                           clear_color=[0.0, 0.0, 0.0, 0.0],
                                           scale=1.0,
                                            )
                    with gr.Row():
                        with gr.Column():
                            latest_image_output = gr.Image(label="Latest Image")
                            # Add new inputs for n_images and save_scene_every_nth
                            n_images_input = gr.Number(label="Number of Images", value=10)
                            save_scene_every_nth_input = gr.Number(label="Save Scene Every Nth Image", value=10)
                        with gr.Column():
                            submit_button = gr.Button("Generate 3D Scene")
                            cancel_button = gr.Button("Cancel")
                            cancel_message = gr.Textbox(label="Status", lines=1, interactive=False)
                            

                # JSON Generator section within the Text2Room tab
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(label="Prompt")
                        fn_name_input = gr.Dropdown(choices=list(function_inputs.keys()), label="Function Name", value="forward")
                        height_input = gr.Number(label="Height", visible=False)
                        rot_input = gr.Number(label="Rotation", visible=False)
                        txmax_input = gr.Number(label="Txmax", visible=False)
                        radius_input = gr.Number(label="Radius", visible=False)
                        phi_input = gr.Number(label="Phi", visible=False)
                        tzmax_input = gr.Number(label="Tzmax", visible=False)
                        adaptive_input = gr.Checkbox(label="Adaptive Radius")
                        function_image = gr.Image(label="Function Preview", visible=True, elem_classes="function-image-small", value=function_inputs["forward"]["image"])

                    with gr.Column():
                        
                        file_name_input = gr.Textbox(label="JSON File Name", value="generated_scene", placeholder="Enter file name without extension")
                        add_button = gr.Button("Add Entry")
                        download_button = gr.Button("Download JSON")
                        json_display = gr.Textbox(label="JSON Output", lines=20, interactive=True)
                        upload_json = gr.File(label="Upload JSON", file_types=[".json"])
                        clear_button = gr.Button("Clear Everything")
                        example_json_files = list_json_files(EXAMPLES_DIR)
                        example_json_selector = gr.Dropdown(label="Select Example JSON", choices=example_json_files)
                        load_example_button = gr.Button("Load Example JSON")

                # Update inputs visibility based on selected function
                fn_name_input.change(fn=update_inputs, inputs=[fn_name_input], 
                                    outputs=[height_input, rot_input, txmax_input, radius_input, phi_input, tzmax_input, function_image])

                # Add entry to JSON
                add_button.click(fn=add_entry, 
                                inputs=[prompt_input, fn_name_input, height_input, rot_input, txmax_input, radius_input, phi_input, tzmax_input, adaptive_input], 
                                outputs=json_display)

                # Load the uploaded JSON file and display it
                upload_json.change(fn=load_json, inputs=[upload_json], outputs=[json_display])

                # Load selected example JSON file into the JSON viewer
                load_example_button.click(fn=load_example_json, inputs=[example_json_selector], outputs=[json_display])

                # Trigger download of JSON file with the specified name
                download_button.click(fn=download_json_with_name, inputs=[json_display, file_name_input], outputs=download_button)

                # Clear everything
                def clear_everything_fn():
                    return "", "forward", None, 0, 0, 0, 0, 0, 0, False, "", None, None

                clear_button.click(
                    fn=clear_everything_fn, 
                    inputs=[], 
                    outputs=[
                        prompt_input, 
                        fn_name_input, 
                        function_image, 
                        height_input, 
                        rot_input, 
                        txmax_input, 
                        radius_input, 
                        phi_input, 
                        tzmax_input, 
                        adaptive_input, 
                        json_display,
                        output_3d,
                        latest_image_output
                    ]
                )

                # Button action for scene generation
                submit_button.click(fn=process_text2room, inputs=[json_display, file_name_input], outputs=[latest_image_output, output_3d, cancel_message])

                cancel_button.click(fn=cancel_run, inputs=[], outputs=cancel_message)


            """   
            # Additional Tab for Video
            with gr.Tab("Generation with Gaussian Splat"):
                gr.Interface(
                    fn=process_image,
                    inputs=[
                        gr.Image(label="Input Image", sources=["upload", "clipboard"], type="pil"),
                        gr.Textbox(label="Scene Hallucination Prompt")
                    ],
                    outputs=gr.Model3D(label="Generated Scene"),
                    allow_flagging="never",
                    title="Generating a Scene",
                    description="Hallucinate 3D scenes",
                    article="",
                    examples=[
                        ["samples/images/magnific-2navRnauqRDOCHjQlFXO-image-1718724248-1.png", "a suburban street in North Carolina on a bright, sunny day"],
                        
                    ]
                )
                """


    # Launch the combined interface
    demo.queue().launch(share=True)

    
    