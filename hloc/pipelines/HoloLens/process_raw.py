import numpy as np
from PIL import Image
import os
from pathlib import Path
import cv2
from tqdm import tqdm
import pytransform3d.transformations as pt


def rotate_and_undistort(directory, rotation, output_directory, prefix, calib_values):
    # Create a new directory to save rotated images
    os.makedirs(output_directory, exist_ok=True)

    # Get a list of all .pgm files in the input directory
    pgm_files = [f for f in os.listdir(directory) if f.lower().endswith('.pgm')]

    fx, fy, cx, cy, k1, k2, p1, p2 = calib_values
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2])
    

    for pgm_file in tqdm(pgm_files[::5]):
        pgm_path = os.path.join(directory, pgm_file)
        # output_path = os.path.join(output_directory, os.path.splitext(pgm_file)[0] + ".jpg")

        # Open and rotate the image
        img = Image.open(pgm_path)
        rotated_img = img.rotate(rotation, expand=True)
        img = np.array(rotated_img)
        # Undistort the image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
        file_path = output_dir / (prefix + "_" + os.path.splitext(pgm_file)[0] + ".jpg")
        cv2.imwrite(str(file_path), undistorted_img)



# fx, fy, cx, cy, k1, k2, p1, p2
calib_dict = {"LL": [368.61479342318705, 367.43916126955804, 238.42596191008707, 313.12567726452966, -0.013393300298381391, 0.011888899638582105, 0.0011904608842684813, -0.00266087031273223],
              "LF": [368.4289947499528, 366.4996978518904, 240.7099494326709, 328.71927435171773, -0.012895389138074139, 0.010034119982097077, 0.00041317110288764853, 0.003647604939504504],
              "RF": [365.60869205550324, 364.353928958462, 238.04518195476317, 307.2496122189609, -0.017645221963269304, 0.01593660856254029, 0.0005756543331265867, -0.004860614369086555],
              "RR": [367.53812625806944, 366.64360419643, 242.53826334243405, 324.9290992485602, -0.01713131653383185, 0.015211171504114605, -4.6716224768668506e-05, 0.0026202035108305725],
              }




data_dir = Path("/local/home/hanlonm/data/HL_captures/DLAB/2023-08-22-113629")

pose_file_name = data_dir / "image_poses.txt"
pose_file = open(pose_file_name, "w")
pose_file.write("# image_file tx ty tz qx qy qz qw \n")

trajectories_file = np.loadtxt(data_dir / "VLC_LF_rig2world.txt", dtype=str, delimiter=",")
for position in trajectories_file:
    ts = position[0]
    T_world_rig: np.ndarray  = position[1:]
    file_name = "mapping/LF_" + ts + ".jpg"
    T_world_rig= T_world_rig.astype(float).reshape((4,4))
    try:
        pq = pt.pq_from_transform(T_world_rig)
        pose_file.write(file_name + " " +
                        str(pq[0]) + " " + str(pq[1]) +
                        " " + str(pq[2]) + " " +
                        str(pq[4]) + " " + str(pq[5]) +
                        " " + str(pq[6]) + " " +
                        str(pq[3]) + "\n")
    except:
        continue

folders = ["VLC_LL", "VLC_LF", "VLC_RF", "VLC_RR"]
output_dir = data_dir / "mapping"
for folder in folders:
    dir = data_dir / folder
    prefix = folder.split("_")[-1]
    if folder in ["VLC_LF", "VLC_RR"]:
        rotate_and_undistort(dir, -90, output_dir, prefix, calib_dict[prefix])
    else:
        rotate_and_undistort(dir, 90, output_dir, prefix, calib_dict[prefix])