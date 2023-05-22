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
import argparse
from pose_utils import compute_absolute_pose_error




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00596")
    parser.add_argument("--run_name", type=str, default="test1")
    args = parser.parse_args()

    home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")
    environment = args.environment
    dataset = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    
    environment_dataset_path = Path(home_dir) / "mt-matthew/data"
    run_id = args.run_name
    hf = h5py.File(str(environment_dataset_path) + f"/training_data/{run_id}.h5", "r+")


    images = dataset
    loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file



    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')

    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))

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
    

    localization_results: dict = logs['loc']
    sorted_result_keys = list(localization_results.keys())
    sorted_result_keys.sort()

    T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                np.array([0, 0.0, 0.0]))

    T_world_map = pt.transform_from(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([0, 0.0, 0.0]))



    gt = hf[environment]["pose_data"][:]
    preds = []
    for result_key in sorted_result_keys:
        result_key: str

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
        #gt.append(pose_data[position][view])

    preds = np.array(preds)
    errors = compute_absolute_pose_error(p_es_aligned=preds[:,:3], q_es_aligned=preds[:,3:],
                                        p_gt=gt[:,:3], q_gt=gt[:,3:])

    e_trans_vec = errors[1]
    e_rot = np.array([errors[2]]).T

    errors = np.hstack((e_trans_vec, e_rot))

    hf[environment].create_dataset("errors", data=errors)
    hf.close()

    # clean up
    fs = h5py.File(local_features, 'r')
    fd = h5py.File(outputs / (feature_conf['output']+'_copy.h5'), 'w')
    for a in fs.attrs:
        fd.attrs[a] = fs.attrs[a]
    for d in fs:
        if not 'training' in d: fs.copy(d, fd)
    fs.close()
    fd.close()
    os.remove(local_features)
    os.rename(outputs / (feature_conf['output']+'_copy.h5'), local_features)


if __name__ == "__main__":
    main()