
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.renderer import PerspectiveCameras

from pytorch3d.vis.plotly_vis import get_camera_wireframe, plot_scene

from pytorch3d.renderer import PerspectiveCameras, join_cameras_as_batch, FoVPerspectiveCameras

import os



def calculate_plot_margin(min_vals, max_vals, margin_factor=0.1):
    """
    Calculate the plot margin based on the minimum and maximum values.
    
    Args:
        min_vals: Minimum values for x, y, z axes.
        max_vals: Maximum values for x, y, z axes.
        margin_factor: Factor to determine the margin size. Default is 0.1 (10%).
        
    Returns:
        plot_margin: The calculated margin for each axis.
    """
    return margin_factor * (max_vals - min_vals)

def calculate_bounds_for_cameras(cameras_list):
    """
    Calculate the min and max bounds for all cameras in the list.

    Args:
        cameras_list: List of camera sets to plot.
    
    Returns:
        min_vals: The minimum bounds across all cameras.
        max_vals: The maximum bounds across all cameras.
        center: The center point of all cameras.
    """
    min_vals = np.array([np.inf, np.inf, np.inf])
    max_vals = np.array([-np.inf, -np.inf, -np.inf])

    for cameras in cameras_list:
        cam_wires_canonical = get_camera_wireframe().to(cameras.device)[None]
        cam_trans = cameras.get_world_to_view_transform().inverse()
        cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)

        cam_min_vals = cam_wires_trans.min(dim=1).values.min(dim=0).values.cpu().numpy()
        cam_max_vals = cam_wires_trans.max(dim=1).values.max(dim=0).values.cpu().numpy()

        min_vals = np.minimum(min_vals, cam_min_vals)
        max_vals = np.maximum(max_vals, cam_max_vals)

    # Calculate the center point
    center = (min_vals + max_vals) / 2

    return min_vals, max_vals, center

def ensure_batched_cameras(cams):
    """
    Ensures that the input `cams` is a batched PerspectiveCameras object.
    """
    if isinstance(cams, list):
        return join_cameras_as_batch(cams)
    elif isinstance(cams, (PerspectiveCameras, FoVPerspectiveCameras)):
        if cams.R.ndim == 3:  # If R has three dimensions, it's already batched
            return cams
        else:
            return cams.extend(batch_size=1)
    else:
        raise ValueError("Input must be either a list of PerspectiveCameras or a single PerspectiveCameras object.")

def plot_cameras(ax, cameras, color: str = "blue", label=None):
    """
    Plots a set of `cameras` objects into the matplotlib axis `ax` with
    color `color`. Optionally annotates each camera with its label.
    If trace_path is True, draws a line connecting the cameras to show the path.
    """
    # Ensure the camera wireframe is on the same device as the cameras
    device = cameras.device
    cam_wires_canonical = get_camera_wireframe().to(device)[None]
    cam_trans = cameras.get_world_to_view_transform().inverse()
    #print(cam_trans)
    
    cam_wires_trans = cam_trans.transform_points(cam_wires_canonical)

    plot_handles = []
    trace_points = []

    for i, wire in enumerate(cam_wires_trans):
        x_, z_, y_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=1)
        plot_handles.append(h)
        if label is not None:
            ax.text(x_[0], y_[0], z_[0], f'{label[i]}', color=color)
    return plot_handles


def plot_camera_scene(cameras_list, labels=None, colors=None, annotate=False,title="Camera Visualization",save_path=None):
    """
    Plots multiple sets of cameras in the same plot.

    Args:
        cameras_list: List of camera sets to plot.
        labels: List of labels corresponding to each camera set. Defaults to "Camera 1", "Camera 2", etc.
        colors: List of colors corresponding to each camera set. If None, applies a gradient from blue to red.
        annotate: If True, annotates each camera with its index.
    """
    if labels is None:
        labels = [f"{i}" for i in range(len(cameras_list))]

    if colors is None:
        # Apply gradient colors from blue to red
        colors = cm.coolwarm(np.linspace(0, 1, len(cameras_list)))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection="3d")
    ax.clear()
    ax.set_title(title)

    plot_handles = []
    all_trace_points = []

    for i, (cameras, color) in enumerate(zip(cameras_list, colors)):
        cameras = ensure_batched_cameras(cameras)
        handle = plot_cameras(ax, cameras, color=color, label=[labels[i]] if annotate else None)
        plot_handles.append((handle[0], labels[i]))

    # Calculate the min and max bounds for all cameras
    min_vals, max_vals, center = calculate_bounds_for_cameras(cameras_list)

    # Calculate the maximum range from the center to the bounds
    max_range = np.max(max_vals - min_vals) / 2.0

    # Apply a scaling factor to make the plot slightly larger
    scale_factor = 1.1
    plot_margin = max_range * scale_factor

    # Set plot limits centered around the center point
    ax.set_xlim3d([center[0] - plot_margin, center[0] + plot_margin])
    ax.set_ylim3d([center[1] - plot_margin, center[1] + plot_margin])
    ax.set_zlim3d([center[2] - plot_margin, center[2] + plot_margin])
    """
    # Set plot limits with a fixed aspect ratio
    max_range = np.max(max_vals - min_vals + 2 * plot_margin)
    ax.set_xlim3d([min_vals[0] - plot_margin[0], min_vals[0] - plot_margin[0] + max_range])
    ax.set_ylim3d([min_vals[1] - plot_margin[1], min_vals[1] - plot_margin[1] + max_range])
    ax.set_zlim3d([min_vals[2] - plot_margin[2], min_vals[2] - plot_margin[2] + max_range])
    
    # Set plot limits slightly larger than the bounds
    ax.set_xlim3d([min_vals[0] - plot_margin[0], max_vals[0] + plot_margin[0]])
    ax.set_ylim3d([min_vals[1] - plot_margin[1], max_vals[1] + plot_margin[1]])
    ax.set_zlim3d([min_vals[2] - plot_margin[2], max_vals[2] + plot_margin[2]])
    
    plot_radius = 3
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([3 - plot_radius, 3 + plot_radius])
    ax.set_zlim3d([-plot_radius, plot_radius])
    """
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    #ax.legend([h for h, _ in plot_handles], [l for _, l in plot_handles], loc="upper center", bbox_to_anchor=(0.5, 0))
    # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    #plt.show()

