import os
import PIL
from datetime import datetime

def create_output_path(base_output_path):
    now_str = datetime.now().strftime('%Y-%m-%d')
    output_path = os.path.join(base_output_path, now_str)
    os.makedirs(output_path, exist_ok=True)
    return output_path

def sanitize_filename(prompt):
    return prompt.replace(' ', '_')[:60]

def save_image(image, prompt, idx, outdir):
    prefix = sanitize_filename(prompt)
    filename = f"{prefix[:70]}_{idx:04d}.png"
    file_out = os.path.join(outdir, filename)
    image.save(file_out)
    return file_out