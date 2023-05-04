from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np
import os
import pickle
import h5py
import json

with open('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_HL/00195_HL_hloc_superpoint+superglue_netvlad20.txt_logs.pkl', 'rb') as f:
    logs = pickle.load(f)

result_dict = {}

hf = h5py.File("/local/home/hanlonm/mt-matthew/00195_HL_10_pose_data.h5", "r")
pose_data = hf["pose_data"]
for position in pose_data.attrs.keys():
    pose_dict: dict = json.loads(pose_data.attrs[position])
    view_dict = {}
    for pq_key in pose_dict.keys():
        pq = pose_dict[pq_key]
        view_dict[pq_key] = {"gt": pq}
    result_dict[position] = view_dict
    


localization_results: dict = logs['loc']
sorted_result_keys = list(localization_results.keys())
sorted_result_keys.sort()

T_cam_base = pt.transform_from(
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
            np.array([0, 0.0, 0.0]))

T_world_map = pt.transform_from(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([0, 0.0, 0.0]))


for result_key in sorted_result_keys:
    result_key: str
    split =result_key.replace("/", "_")
    split = split.replace(".", "_")
    split = split.split("_")
    position = str(int(split[1]))
    view = str(int(split[2]))

    result = localization_results[result_key]
    success = result['PnP_ret']['success']
    if success:
        tvec = result['PnP_ret']['tvec']
        qvec = result['PnP_ret']['qvec']
        num_points_detected = len(result['points3D_ids'])
        num_pnp_inliers = result['PnP_ret']['num_inliers']
    else:
        print("Failed to localize image {} !".format(result_key))
        tvec = [0,0,0]
        qvec = [0,0,0,0]
        num_points_detected = 0
        num_pnp_inliers = 0
    rot = pr.matrix_from_quaternion(qvec)
    T_cam_world = pt.transform_from(rot, tvec)

    T_map_base = np.linalg.inv(T_cam_world) @ T_cam_base
    T_world_base = T_world_map @ T_map_base

    pq = pt.pq_from_transform(T_world_base).tolist()

    result_dict[position][view]["pred"] = pq

print()
# Evaluation Paths

# trajectory_dirs = os.listdir(run_path)


# T_cam_base = pt.transform_from(
#             np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
#             np.array([0, 0.0, 0.0]))

# T_world_map = pt.transform_from(
#             np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
#             np.array([0, 0.0, 0.0]))

# for trajectory_dir in trajectory_dirs:
#     trajectory_results = [key for key in sorted_result_keys if trajectory_dir in key]
#     trajectory_variations = os.listdir( run_path / trajectory_dir)

#     for variation in trajectory_variations:
#         variation_results = [key for key in trajectory_results if variation[:-4] in key]
#         pose_estimates = np.zeros((len(variation_results), 7))
        # pose_file = open(run_path/ trajectory_dir / (variation[:-4] + '_loc_est.txt'), "w")
        # pose_file.write("# tx ty tz qw qx qy qz points_detected num_PnP_inliers")
        # for image_result_key in variation_results:
        #     result = localization_results[image_result_key]
        #     success = result['PnP_ret']['success']
        #     if success:
        #         tvec = result['PnP_ret']['tvec']
        #         qvec = result['PnP_ret']['qvec']
        #         num_points_detected = len(result['points3D_ids'])
        #         num_pnp_inliers = result['PnP_ret']['num_inliers']
        #     else:
        #         print("Failed to localize image {} !".format(image_result_key))
        #         tvec = [0,0,0]
        #         qvec = [0,0,0,0]
        #         num_points_detected = 0
        #         num_pnp_inliers = 0
        #     rot = pr.matrix_from_quaternion(qvec)
        #     T_cam_world = pt.transform_from(rot, tvec)

        #     T_map_base = np.linalg.inv(T_cam_world) @ T_cam_base
        #     T_world_base = T_world_map @ T_map_base

        #     pq = pt.pq_from_transform(T_world_base)
        #     loc_stats = np.append(pq, [num_points_detected, num_pnp_inliers])

        #     pose_file.write("\n" + " ".join(map(str, loc_stats)))
        # pose_file.close()    