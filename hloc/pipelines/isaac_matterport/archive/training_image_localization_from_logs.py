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
from pose_utils import compute_absolute_pose_error

with open('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_HL_SPA_NN/00195_HL_SPA_NN_hloc_superpoint+superglue_netvlad20.txt_logs.pkl', 'rb') as f:
    logs = pickle.load(f)

result_dict = {}

hf = h5py.File("/local/home/hanlonm/mt-matthew/data/00195_HL_SPA_NN/test-2000.h5", "r+")
num_points = hf.attrs["num_points"]
num_angles = hf.attrs["num_angles"]

pose_data = hf["pose_data"][:]
errors = np.empty((num_points, num_angles, 4))
    


localization_results: dict = logs['loc']
sorted_result_keys = list(localization_results.keys())
sorted_result_keys.sort()

T_cam_base = pt.transform_from(
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
            np.array([0, 0.0, 0.0]))

T_world_map = pt.transform_from(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([0, 0.0, 0.0]))



gt = []
preds = []
for result_key in sorted_result_keys:
    result_key: str
    split =result_key.replace("/", "_")
    split = split.replace(".", "_")
    split = split.split("_")
    position = int(split[2])
    view = int(split[3])

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

    preds.append( pt.pq_from_transform(T_world_base))
    gt.append(pose_data[position][view])

preds = np.array(preds)
gt = np.array(gt)
errors = compute_absolute_pose_error(p_es_aligned=preds[:,:3], q_es_aligned=preds[:,3:],
                                     p_gt=gt[:,:3], q_gt=gt[:,3:])

e_trans_vec = errors[1]
e_rot = np.array([errors[2]]).T

errors = np.hstack((e_trans_vec, e_rot))
errors = errors.reshape((num_points, num_angles, 4))

print()

hf.create_dataset("errors", data=errors)
hf.close()
