# My Deep Learning Project

This is a deep learning project that includes models for text-to-image, inpainting, ControlNet, and 3D generation.

## Project Structure

- `external/`: Contains external repositories cloned from GitHub.
- `src/`: Contains the main application code.
  - `models/`: Wrappers and interfaces for different models.
  - `utils/`: Utility functions.
  - `main.py`: Entry point of the application.
- `requirements.txt`: Python dependencies.
- `setup.py`: Setup script for packaging.
- `README.md`: Project documentation.

## Setup Instructions

1. Clone the external repositories into the `external/` directory.
2. Install the dependencies using `pip install -r requirements.txt`.
3. Run the application using `python src/main.py`.

# pytorch 
pip install torch==2.2.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
# pytorch 3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install gradio==4.31.4
pip install gradio-litmodel3d==0.0.1
conda install -c conda-forge tbb=2021.6.0

'''
ImportError: /home/descobar/miniconda3/envs/t2room/lib/python3.9/site-packages/pymeshlab/lib/libmeshlab-common.so: undefined symbol: _ZdlPvm, version Qt_5
fix error in wsl ubuntu with this
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/t2room/lib/python3.9/site-packages/pymeshlab/lib:$LD_LIBRARY_PATH
'''
# gsplat
pip install git+https://github.com/nerfstudio-project/gsplat.git

# externals
# using StableFast3d 
https://github.com/Stability-AI/stable-fast-3d.git
# using Text2room
Text2room_DE
# using EscherNet

#using InvisibleStitch
