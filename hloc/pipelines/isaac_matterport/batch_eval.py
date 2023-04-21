from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np
import os


environment = '00195'
dataset = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
images = dataset
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'_loc')
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file
local_features = outputs / 'feats-superpoint-n4096-rmax1600.h5'

# Evaluation Paths
run_path = Path('/local/home/hanlonm/mt-matthew/eval_results/run_1')

feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']
retrieval_conf = extract_features.confs['netvlad']

reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_loc/sfm_superpoint+superglue")

query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'localization/').iterdir()]

features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20, query_prefix="localization",db_prefix="mapping")
loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


# eval_images = np.loadtxt(dataset / 'stamped_groundtruth.txt', int, skiprows=1)[:,0]
query_file = open(dataset / 'queries.txt', "w")
for eval_image in query_image_list:
    query_file.write("{} PINHOLE 1280 720 1462.857177734375 1080.0 640 360 \n".format(str(eval_image)))
query_file.close()

logs = localize_sfm.main(
    reconstruction,
    dataset / 'queries.txt',
    loc_pairs,
    features,
    loc_matches,
    results,
    covisibility_clustering=False)  # not required with SuperPoint+SuperGlue


T_cam_base = pt.transform_from(
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
            np.array([0, 0.0, 0.0]))

T_world_map = pt.transform_from(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array([0, 0.0, 1.8]))

localization_results: dict = logs['loc']
sorted_result_keys = list(localization_results.keys())
sorted_result_keys.sort()

trajectory_dirs = os.listdir(run_path)

for trajectory_dir in trajectory_dirs:
    trajectory_results = [key for key in sorted_result_keys if trajectory_dir in key]
    trajectory_variations = os.listdir( run_path / trajectory_dir)

    for variation in trajectory_variations:
        variation_results = [key for key in trajectory_results if variation[:-4] in key]
        pose_estimates = np.zeros((len(variation_results), 7))
        pose_file = open(run_path/ trajectory_dir / (variation[:-4] + '_loc_est.txt'), "w")
        pose_file.write("# tx ty tz qw qx qy qz points_detected num_PnP_inliers")
        for image_result_key in variation_results:
            result = localization_results[image_result_key]
            tvec = result['PnP_ret']['tvec']
            qvec = result['PnP_ret']['qvec']
            rot = pr.matrix_from_quaternion(qvec)
            T_cam_world = pt.transform_from(rot, tvec)

            T_map_base = np.linalg.inv(T_cam_world) @ T_cam_base
            T_world_base = T_world_map @ T_map_base

            pq = pt.pq_from_transform(T_world_base)
            loc_stats = np.append(pq, [len(result['points3D_ids']), result['PnP_ret']['num_inliers']])

            pose_file.write("\n" + " ".join(map(str, loc_stats)))
        pose_file.close()    
            


    



print(type(localization_results))

