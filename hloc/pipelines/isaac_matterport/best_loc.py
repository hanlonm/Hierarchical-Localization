from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
import matplotlib.pyplot as plt
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
import numpy as np
import os
import argparse
import h5py
import pickle
from pose_utils import compute_absolute_pose_error




def cleanup_h5(original_file, temp_file, remove_key):
    fs = h5py.File(original_file, 'r')
    fd = h5py.File(temp_file, 'w')
    for a in fs.attrs:
        fd.attrs[a] = fs.attrs[a]
    for d in fs:
        if not remove_key in d: fs.copy(d, fd)
    fs.close()
    fd.close()
    os.remove(original_file)
    os.rename(temp_file, original_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="00195")
    parser.add_argument("--run_name", type=str, default="test")
    args = parser.parse_args()

    home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")
    environment = args.environment
    eval_dir = Path(home_dir) / "mt-matthew/data/{}/best_loc".format(
        environment)
    
    dataset = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    images = dataset
    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')
    global_features = outputs / (retrieval_conf['output']+'.h5')

    cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="best_loc")
    cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="best_loc")

    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))
    query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'best_loc/').iterdir()]

    features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, pairs, num_matched=20, query_prefix="best_loc",db_prefix="mapping")
    loc_matches = match_features.main(matcher_conf, pairs, feature_conf['output'], outputs)


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
        pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False)  # not required with SuperPoint+SuperGlue
    
    localization_results: dict = logs['loc']

    T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                np.array([0, 0.0, 0.0]))

    T_world_map = pt.transform_from(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([0, 0.0, 0.0]))
    
    localization_results: dict = logs['loc']
    sorted_result_keys = list(localization_results.keys())
    sorted_result_keys.sort()
    print()

    path_file = np.load(eval_dir / "paths.npz")
    path_keys = list(path_file.keys())
    path_keys = [str(key) for key in path_keys]

    save_localizations_dict = {}
    for path_key in path_keys:
        path_poses = path_file[path_key]
        path_results = [result_key for result_key in sorted_result_keys if path_key in result_key]
        len_results = len(path_results)
        it = 0
        best_path = []
        pose_file = open(eval_dir/  (path_key + '_loc_est.txt'), "w")
        pose_file.write("# tx ty tz qw qx qy qz points_detected num_PnP_inliers")
        path_localizations = []
        for i, waypoint in enumerate(path_poses):
            preds = []
            gt_poses = []
            for j, view in enumerate(waypoint):
                gt_poses.append(view[0])

                result = localization_results[path_results[it]]
                success = result['PnP_ret']['success']
                if success:
                    tvec = result['PnP_ret']['tvec']
                    qvec = result['PnP_ret']['qvec']
                    num_points_detected = len(result['points3D_ids'])
                    num_pnp_inliers = result['PnP_ret']['num_inliers']
                else:
                    print("Failed to localize image {} !".format(path_results[it]))
                    tvec = [0,0,0]
                    qvec = [0,0,0,0]
                    num_points_detected = 0
                    num_pnp_inliers = 0
                rot = pr.matrix_from_quaternion(qvec)
                T_cam_world = pt.transform_from(rot, tvec)

                T_map_base = np.linalg.inv(T_cam_world) @ T_cam_base
                T_world_base = T_world_map @ T_map_base

                pq = pt.pq_from_transform(T_world_base)
                pq = np.append(pq, [num_points_detected, num_pnp_inliers])
                preds.append(pq)
                it += 1
            preds = np.array(preds)
            path_localizations.append(preds)

            errors = compute_absolute_pose_error(p_es_aligned=preds[:,:3], q_es_aligned=preds[:,3:-2],
                                        p_gt=waypoint[:,0,:3], q_gt=waypoint[:,0,3:])
            
            e_trans = errors[0]
            e_trans_vec = errors[1]
            e_rot = np.array([errors[2]]).T

            best_view = waypoint[np.argmin(e_trans)]
            best_pred = preds[np.argmin(e_trans)]
            pose_file.write("\n" + " ".join(map(str, best_pred)))
            best_path.append(best_view)

        path_localizations = np.array(path_localizations)
        save_localizations_dict[path_key] = path_localizations
        
        pose_file.close()
        best_path = np.array(best_path)
        np.save(eval_dir/(path_key + '.npy'), best_path)

    np.savez(eval_dir / "localizations.npz", **save_localizations_dict)


            

                


if __name__ == "__main__":
    
    main()

