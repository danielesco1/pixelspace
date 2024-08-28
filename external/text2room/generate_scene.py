import os
import json
from PIL import Image

from model.text2room_pipeline import Text2RoomPipeline
from model.utils.opt import get_default_parser
from model.utils.utils import save_poisson_mesh, generate_first_image

import torch

@torch.no_grad()
def main(args):
    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))

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