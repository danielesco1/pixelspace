import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)

import plotly.graph_objects as go
import trimesh
import numpy as np

def visualize_mesh(output_file):
    # Load the mesh using trimesh
    scene = trimesh.load(output_file)
    
    if isinstance(scene, trimesh.Scene):
        # Extract all meshes from the scene
        meshes = scene.geometry.values()
    else:
        meshes = [scene]  # Single mesh
    
    # Create a Plotly figure
    fig = go.Figure()

    # Iterate over each mesh in the scene
    for mesh in meshes:
        # Extract vertex colors if available, else create a default color
        if mesh.visual.kind == 'vertex' and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors[:, :3] / 255.0  # Convert to [0, 1] range
        else:
            # Create a default color for each vertex if no vertex colors are available
            vertex_colors = np.tile([0.5, 0.5, 0.9], (mesh.vertices.shape[0], 1))  # Example: light blue

        # Add the mesh to the Plotly figure
        fig.add_trace(go.Mesh3d(
            x=mesh.vertices[:, 0],
            y=mesh.vertices[:, 1],
            z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            vertexcolor=vertex_colors,
            opacity=0.5
        ))

    # Show the plot
    fig.show()
    

def visualize_glb(output_file):
    # Load the mesh using PyTorch3D's `load_objs_as_meshes` function
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Since PyTorch3D does not directly load .glb files, convert to .obj or use trimesh as an intermediary
    mesh = load_objs_as_meshes([output_file], device=device)

    # Initialize the camera
    R, T = look_at_view_transform(dist=2.7, elev=10, azim=150)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # Define the rasterization and shading settings
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Place a point light in the scene
    lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

    # Create a phong renderer by composing a rasterizer and a shader
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    # Render the mesh
    images = renderer(mesh)

    # Convert images to a numpy array and visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()
