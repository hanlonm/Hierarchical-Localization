from PIL import Image
import os
from pathlib import Path

def rotate_images(directory, rotation, output_directory):
    # Create a new directory to save rotated images
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all .pgm files in the input directory
    pgm_files = [f for f in os.listdir(directory) if f.lower().endswith('.pgm')]

    for pgm_file in pgm_files:
        pgm_path = os.path.join(directory, pgm_file)
        output_path = os.path.join(output_directory, os.path.splitext(pgm_file)[0] + ".jpg")

        # Open and rotate the image
        img = Image.open(pgm_path)
        rotated_img = img.rotate(rotation, expand=True)

        # Save the rotated image as a .jpg file
        rotated_img.save(output_path, "JPEG")

    print(f"Rotated images saved in {output_directory}")

data_dir = Path("/local/home/hanlonm/data/HL_captures/calibration_cvg_5/2023-08-16-180750")
folders = ["VLC_LL", "VLC_LF", "VLC_RF", "VLC_RR"]

for folder in folders:
    dir = data_dir / folder
    output_dir = data_dir / folder.split("_")[-1]
    if folder in ["VLC_LF", "VLC_RR"]:
        rotate_images(dir, -90, output_dir)
    else:
        rotate_images(dir, 90, output_dir)






