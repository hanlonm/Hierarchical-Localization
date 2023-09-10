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
    parser.add_argument("--run_name", type=str, default="best_test")
    args = parser.parse_args()

    home_dir = os.environ.get("BASE_DIR", "/local/home/hanlonm")
    environment = args.environment
    dataset = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    images = dataset
    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file

    
    # Evaluation Paths
    run_name = args.run_name
    run_path = Path(home_dir + '/active-viewpoint-selection/eval_results/path_data') / str(f"{environment}_{run_name}")

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')
    global_features = outputs / (retrieval_conf['output']+'.h5')

    cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="localization")
    cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="localization")


    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))

    query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'localization/').iterdir()]

    features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, loc_pairs, num_matched=20, query_prefix="localization",db_prefix="mapping")
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


    T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                np.array([0, 0.0, 0.0]))

    T_world_map = pt.transform_from(
                np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.array([0, 0.0, 0.0]))

    localization_results: dict = logs['loc']
    sorted_result_keys = list(localization_results.keys())
    sorted_result_keys.sort()

    trajectory_dirs = os.listdir(run_path)
    trajectory_dirs = [dir for dir in trajectory_dirs if "best_loc" not in dir]

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
                success = result['PnP_ret']['success']
                if success:
                    tvec = result['PnP_ret']['tvec']
                    qvec = result['PnP_ret']['qvec']
                    num_points_detected = len(result['points3D_ids'])
                    num_pnp_inliers = result['PnP_ret']['num_inliers']
                else:
                    print("Failed to localize image {} !".format(image_result_key))
                    tvec = [0,0,0]
                    qvec = [0,0,0,0]
                    num_points_detected = 0
                    num_pnp_inliers = 0
                rot = pr.matrix_from_quaternion(qvec)
                T_cam_world = pt.transform_from(rot, tvec)

                T_map_base = np.linalg.inv(T_cam_world) @ T_cam_base
                T_world_base = T_world_map @ T_map_base

                pq = pt.pq_from_transform(T_world_base)
                loc_stats = np.append(pq, [num_points_detected, num_pnp_inliers])

                pose_file.write("\n" + " ".join(map(str, loc_stats)))
            pose_file.close()    

    # clean up
    cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="localization")
    cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="localization")

            

if __name__ == "__main__":
    
    main()

