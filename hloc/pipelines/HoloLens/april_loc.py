import cv2
from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization
import pycolmap
from pytransform3d import transformations as pt
from pytransform3d import rotations as pr
from pytransform3d.transform_manager import TransformManager 
import argparse
import h5py
import numpy as np
import os
from typing import List
import matplotlib.pyplot as plt


from dt_apriltags import Detector, Detection
from sksurgerycore.algorithms.averagequaternions import average_quaternions

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
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DLAB_6")
    args = parser.parse_args()
    home_dir = os.environ.get("BASE_DIR", "/local/home/hanlonm")
    environment = args.environment
    dataset = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    images = dataset

    fx,fy,cx,cy = 4276.911590668629, 4270.889749599398, 3079.736876563767, 2001.9942184986703
    k1, k2, p1, p2 = -0.14810513553665675, 0.10954548001908843, -0.0021822466431448665, 0.000529327533272714
    # Load the camera intrinsic and distortion parameters
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2])
    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)

    camera_params = (fx,fy,cx,cy)
    camera_string = f"OPENCV 6240 4160 {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}"

    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')
    global_features = outputs / (retrieval_conf['output']+'.h5')

    cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="tag_loc")
    cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="tag_loc")

    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))
    query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'tag_loc/').iterdir()]

    features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, pairs, num_matched=20, query_prefix="tag_loc",db_prefix="mapping")
    loc_matches = match_features.main(matcher_conf, pairs, feature_conf['output'], outputs)

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

    T_y_up = pt.transform_from(
                np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
                np.array([0, 0.0, 0.0]))
    

    tm = TransformManager()
    translations = []
    quats = []
    for i, image in enumerate(query_image_list):
        result = localization_results[image]["PnP_ret"]
        if result["success"]:
            tvec = result['tvec']
            qvec = result['qvec']
            rot = pr.matrix_from_quaternion(qvec)
            T_cam_world = pt.transform_from(rot, tvec)

            img = cv2.imread(str(dataset/image))

            # Undistort the image
            undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
            detect_image = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

            tags: List[Detection] = at_detector.detect(detect_image, True, camera_params, 0.1365)
            if len(tags) > 1:
                print("too many tags in image!")
                continue
            tag: Detection = tags[0]
            tag_R = tag.pose_R
            tag_t = tag.pose_t.flatten()
            T_cam_tag = pt.transform_from(tag_R, tag_t)

            T_world_tag = pt.invert_transform(T_cam_world) @ T_cam_tag
            pq = pt.pq_from_transform(T_world_tag)
            translations.append(pq[:3])
            quats.append(pq[3:])

            # tm.add_transform(f"tag_{i}", "world", T_world_tag)

            print(T_world_tag)
    translations = np.array(translations)
    quats = np.array(quats)
    q = average_quaternions(quaternions=quats[[0,1,2,4,5,6]])
    p = np.mean(translations[[0,1,2,4,5,6]], axis=0)
    pq = np.concatenate((p,q))
    T_world_tag_avg = pt.transform_from_pq(pq)
    T_world_tag_avg[:3, :3] =  T_world_tag_avg[:3, :3] @ T_y_up[:3, :3] 
    T_tag_world = pt.invert_transform(T_world_tag_avg)

    tm.add_transform(f"loc_tag", "map", T_world_tag_avg)
    tm.plot_frames_in("map", show_name=True, s=2)
    print()
    print(T_world_tag_avg)

    print("Use this pq for ROS map->loc_tag")
    print(pt.pq_from_transform(T_world_tag_avg))
    plt.show()


if __name__ == "__main__":
    
    main()

