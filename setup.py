import os

def create_dir_structure():
    # Define the root directory
    root_dir = 'project-root'

    # Define the subdirectories
    directories = [
        os.path.join(root_dir, 'external'),
        os.path.join(root_dir, 'external', 'model_repo_1'),
        os.path.join(root_dir, 'external', 'model_repo_2'),
        os.path.join(root_dir, 'src'),
        os.path.join(root_dir, 'src', 'models'),
        os.path.join(root_dir, 'src', 'utils'),
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create empty files
    files = [
        os.path.join(root_dir, 'src', '__init__.py'),
        os.path.join(root_dir, 'src', 'main.py'),
        os.path.join(root_dir, 'src', 'models', 'model_1_wrapper.py'),
        os.path.join(root_dir, 'src', 'models', 'model_2_wrapper.py'),
        os.path.join(root_dir, 'src', 'utils', 'utils.py'),
        os.path.join(root_dir, 'requirements.txt'),
        os.path.join(root_dir, 'README.md'),
        os.path.join(root_dir, 'setup.py'),
    ]

    for file in files:
        with open(file, 'w') as f:
            f.write('')
        print(f"Created file: {file}")

    # Example content for setup.py
    setup_content = """
from setuptools import setup, find_packages

setup(
    name='my_project',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'my_project=src.main:main',
        ],
    },
)
    """
    with open(os.path.join(root_dir, 'setup.py'), 'w') as f:
        f.write(setup_content.strip())
    print(f"Written setup.py content")

    # Example content for README.md
    readme_content = """
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
    """
    with open(os.path.join(root_dir, 'README.md'), 'w') as f:
        f.write(readme_content.strip())
    print(f"Written README.md content")


if __name__ == '__main__':
    create_dir_structure()
