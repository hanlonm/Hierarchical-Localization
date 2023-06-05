from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np



environment = '00195'
dataset = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
images = dataset
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'_loc')
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file
local_features = outputs / 'feats-superpoint-n4096-rmax1600.h5'



feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']
retrieval_conf = extract_features.confs['netvlad']

reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_loc/sfm_superpoint+superglue")

query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'localization/').iterdir()]

features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

global_descriptors = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20, query_prefix="localization",db_prefix="mapping")
loc_matches = match_features.main(matcher_conf, loc_pairs, feature_conf['output'], outputs)


eval_images = np.loadtxt(dataset / 'stamped_groundtruth.txt', int, skiprows=1)[:,0]
query_file = open(dataset / 'queries.txt', "w")
for eval_image in eval_images:
    query_file.write("localization/{}.jpeg PINHOLE 1280 720 1462.857177734375 1080.0 640 360 \n".format(str(eval_image)))
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

localization_results: dict = logs['loc']


pose_file = open(dataset / "stamped_traj_estimate.txt", "w")
pose_file.write("# timestamp tx ty tz qx qy qz qw \n")

for i, (name, result) in enumerate(localization_results.items()):
    print(name)
    tvec = result['PnP_ret']['tvec']
    qvec = result['PnP_ret']['qvec']
    #qvec =  qvec[[1,2,3,0]]
    rot = pr.matrix_from_quaternion(qvec)
    T_cam_world = pt.transform_from(rot, tvec)

    T_world_base = np.linalg.inv(T_cam_world) @ T_cam_base

    pq = pt.pq_from_transform(T_world_base)
    pq = pq[[0,1,2,4,5,6,3]]
    print(list(pq))
    pose_file.write(
            str(eval_images[i])
            + " "
            + str(pq[0])
            + " "
            + str(pq[1])
            + " "
            + str(pq[2])
            + " "
            + str(pq[3])
            + " "
            + str(pq[4])
            + " "
            + str(pq[5])
            + " "
            + str(pq[6])
            + "\n"
        )
    
pose_file.close()

print(type(localization_results))

