from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np
import os
import h5py
import json


environment = '00195_HL'
dataset = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
images = dataset
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment)
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file



feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['NN-superpoint']
retrieval_conf = extract_features.confs['netvlad']

local_features = outputs / (feature_conf['output']+'.h5')

reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))

query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'training/').iterdir()]

features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20, query_prefix="training",db_prefix="mapping")
loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


# eval_images = np.loadtxt(dataset / 'stamped_groundtruth.txt', int, skiprows=1)[:,0]
# spot_cam K: [609.5238037109375, 0.0, 640.0, 0.0, 610.1694946289062, 360.0, 0.0, 0.0, 1.0] 
# camera_string = "PINHOLE 1280 720 1462.857177734375 1080.0 640 360"
camera_string = "PINHOLE 1280 720 609.5238037109375 610.1694946289062 640 360"
query_file = open(dataset / 'queries.txt', "w")
for eval_image in query_image_list:
    query_file.write("{} {} \n".format(str(eval_image), camera_string))
query_file.close()

logs = localize_sfm.main(
    reconstruction,
    dataset / 'queries.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue


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