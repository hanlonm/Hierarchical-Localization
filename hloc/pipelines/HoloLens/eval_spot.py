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
import shutil
from typing import List
import matplotlib.pyplot as plt
import json

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

def logmap_so3(R):
    """Logmap at the identity.
    Returns canonical coordinates of rotation.
    cfo, 2015/08/13

    """
    R11 = R[0, 0]
    R12 = R[0, 1]
    R13 = R[0, 2]
    R21 = R[1, 0]
    R22 = R[1, 1]
    R23 = R[1, 2]
    R31 = R[2, 0]
    R32 = R[2, 1]
    R33 = R[2, 2]
    tr = np.trace(R)
    omega = np.empty((3, ), dtype=np.float64)

    # when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, we do something
    # special
    if (np.abs(tr + 1.0) < 1e-10):
        if (np.abs(R33 + 1.0) > 1e-10):
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R33)) * \
                np.array([R13, R23, 1.0+R33])
        elif (np.abs(R22 + 1.0) > 1e-10):
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R22)) * \
                np.array([R12, 1.0+R22, R32])
        else:
            omega = (np.pi / np.sqrt(2.0 + 2.0 * R11)) * \
                np.array([1.0+R11, R21, R31])
    else:
        magnitude = 1.0
        tr_3 = tr - 3.0
        if tr_3 < -1e-7:
            theta = np.arccos((tr - 1.0) / 2.0)
            magnitude = theta / (2.0 * np.sin(theta))
        else:
            # when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            # use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            magnitude = 0.5 - tr_3 * tr_3 / 12.0

        omega = magnitude * np.array([R32 - R23, R13 - R31, R21 - R12])

    return omega

def compute_absolute_pose_error(p_es_aligned, q_es_aligned, p_gt, q_gt):
    e_trans_vec = (p_gt - p_es_aligned)
    e_trans = np.sqrt(np.sum(e_trans_vec**2, 1))

    # orientation error
    e_rot = np.zeros((len(e_trans, )))
    e_ypr = np.zeros(np.shape(p_es_aligned))
    for i in range(np.shape(p_es_aligned)[0]):
        R_we = pr.matrix_from_quaternion(q_es_aligned[i, :])
        R_wg = pr.matrix_from_quaternion(q_gt[i, :])
        e_R = np.dot(R_we, np.linalg.inv(R_wg))
        e_ypr[i, :] = pr.euler_from_matrix(e_R, 2, 1, 0, False)
        e_rot[i] = np.rad2deg(np.linalg.norm(logmap_so3(e_R[:3, :3])))
        pt.transform_log_from_transform

    # scale drift
    motion_gt = np.diff(p_gt, 0)
    motion_es = np.diff(p_es_aligned, 0)
    dist_gt = np.sqrt(np.sum(np.multiply(motion_gt, motion_gt), 1))
    dist_es = np.sqrt(np.sum(np.multiply(motion_es, motion_es), 1))
    e_scale_perc = np.abs((np.divide(dist_es, dist_gt) - 1.0) * 100)

    return e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc

def gt_body_loc(result_dict, dataset, experiment_dir: Path, home_dir, environment, images):
    if os.path.exists(dataset/"tag"):
        shutil.rmtree(dataset/"tag")

    shutil.copytree(experiment_dir/"tag", dataset/"tag")

    dslr_fx,dslr_fy,dslr_cx,dslr_cy = 4276.911590668629, 4270.889749599398, 3079.736876563767, 2001.9942184986703
    dslr_k1, dslr_k2, dslr_p1, dslr_p2 = -0.14810513553665675, 0.10954548001908843, -0.0021822466431448665, 0.000529327533272714
    # Load the camera intrinsic and distortion parameters
    dslr_camera_matrix = np.array([[dslr_fx, 0, dslr_cx],
                            [0, dslr_fy, dslr_cy],
                            [0, 0, 1]])
    dslr_dist_coeffs = np.array([dslr_k1, dslr_k2, dslr_p1, dslr_p2])

    at_detector = Detector(families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
    
    dslr_camera_params = (dslr_fx,dslr_fy,dslr_cx,dslr_cy)
    dslr_camera_string = f"OPENCV 6240 4160 {dslr_fx} {dslr_fy} {dslr_cx} {dslr_cy} {dslr_k1} {dslr_k2} {dslr_p1} {dslr_p2}"

    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')
    global_features = outputs / (retrieval_conf['output']+'.h5')

    # cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="tag")
    # cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="tag")

    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))
    query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'tag/').iterdir()]

    features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, pairs, num_matched=20, query_prefix="tag",db_prefix="mapping")
    loc_matches = match_features.main(matcher_conf, pairs, feature_conf['output'], outputs)

    query_file = open(dataset / 'queries.txt', "w")
    for eval_image in query_image_list:
        query_file.write("{} {} \n".format(str(eval_image), dslr_camera_string))
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
    gt_pose_file_path = experiment_dir/ "gt/pose_file.txt"
    gt_pose_dict = {}
    with open(gt_pose_file_path) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            if line[0] == '#' or line == '':
                continue
            data = line.split()
            tf_name= data[0]
            pq = np.array([float(val) for val in data[1:]])
            gt_pose_dict[tf_name] = pq

    tm.add_transform("loc_tag", "cam", pt.transform_from_pq(gt_pose_dict["tf_cam_tag"]))
    tm.add_transform("body_gt", "cam", pt.transform_from_pq(gt_pose_dict["tf_cam_body"]))
    tm.add_transform("body_tagloc", "map", pt.transform_from_pq(gt_pose_dict["tf_map_body"]))
    T_tag_body = tm.get_transform("body_gt", "loc_tag")

    pq_map_bodytag = gt_pose_dict["tf_map_body"].reshape((1, 7))
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
            undistorted_img = cv2.undistort(img, dslr_camera_matrix, dslr_dist_coeffs)
            detect_image = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

            tags: List[Detection] = at_detector.detect(detect_image, True, dslr_camera_params, 0.1365)
            tags = [tag for tag in tags if tag.tag_id==2]
            if len(tags) != 1:
                print("too many tags in image!")
                continue
            tag: Detection = tags[0]
            tag_R = tag.pose_R
            tag_t = tag.pose_t.flatten()
            T_cam_tag = pt.transform_from(tag_R, tag_t)

            T_world_tag = pt.invert_transform(T_cam_world) @ T_cam_tag
            pq = pt.pq_from_transform(T_world_tag)
            T_world_tag[:3, :3] =  T_world_tag[:3, :3] @ T_y_up[:3, :3]

            tm.add_transform(f"loc_tag", "map", T_world_tag)
            pq_map_bodygt = pt.pq_from_transform(tm.get_transform("body_gt", "map")).reshape((1, 7))
    
            e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_pose_error(
                pq_map_bodygt[:, :3], pq_map_bodygt[:, 3:7], pq_map_bodytag[:, :3], pq_map_bodytag[:, 3:])

            # tm.add_transform(f"tag_{i}", "world", T_world_tag)

            print(T_world_tag)
            if e_trans<1.0 and (T_world_tag[2,3] > -0.62) and (T_world_tag[2,3]<-0.5):
                
                translations.append(pq[:3])
                quats.append(pq[3:])
            else:
                print("BAD LOC")
                print(e_trans)
                print(pq_map_bodygt)
                print(pq_map_bodytag)
    translations = np.array(translations)
    quats = np.array(quats)
    q = average_quaternions(quaternions=quats[:])
    p = np.mean(translations[:], axis=0)
    pq = np.concatenate((p,q))
    T_world_tag_avg = pt.transform_from_pq(pq)
    T_world_tag_avg[:3, :3] =  T_world_tag_avg[:3, :3] @ T_y_up[:3, :3] 
    T_tag_world = pt.invert_transform(T_world_tag_avg)

    tm.add_transform(f"loc_tag", "map", T_world_tag_avg)
    tm.plot_frames_in("map", show_name=True, s=1)

    pq_map_bodygt = pt.pq_from_transform(tm.get_transform("body_gt", "map")).reshape((1, 7))
    
    e_trans, e_trans_vec, e_rot, e_ypr, e_scale_perc = compute_absolute_pose_error(
                pq_map_bodygt[:, :3], pq_map_bodygt[:, 3:7], pq_map_bodytag[:, :3], pq_map_bodytag[:, 3:])
    
    if e_trans>0.5:
        print("WARNING POSSIBLE TAG LOCALIZATION FAILURE!")
        #plt.show()
    result_dict[experiment_dir.name] = {"gt": pq_map_bodygt.tolist()}

    return result_dict


def method_body_loc(result_dict, dataset, experiment_dir: Path, home_dir, environment, images):
    camera_string = "PINHOLE 640 480 552.0291012161067 552.0291012161067 320 240"

    outputs = Path(home_dir +'/Hierarchical-Localization/outputs/'+environment)
    pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD
    results = outputs / (environment + '_hloc_superpoint+superglue_netvlad20.txt')  # the result file

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']
    retrieval_conf = extract_features.confs['netvlad']

    local_features = outputs / (feature_conf['output']+'.h5')
    global_features = outputs / (retrieval_conf['output']+'.h5')

    # cleanup_h5(original_file=local_features, temp_file=outputs / (feature_conf['output']+'_copy.h5'), remove_key="spot_eval")
    # cleanup_h5(original_file=global_features, temp_file=outputs / (retrieval_conf['output']+'_copy.h5'), remove_key="spot_eval")

    reconstruction = pycolmap.Reconstruction(home_dir + "/Hierarchical-Localization/outputs/{}/reconstruction".format(environment))
    query_image_list = [p.relative_to(dataset).as_posix() for p in (dataset / 'spot_eval/').iterdir()]

    features = extract_features.main(feature_conf, images, outputs, image_list=query_image_list, feature_path=local_features, overwrite=False)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(global_descriptors, pairs, num_matched=20, query_prefix="spot_eval",db_prefix="mapping")
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

    for i, image in enumerate(query_image_list):
        image: str
        full_name = image.split("/")[1]
        exp = full_name[:17]
        method = full_name[18:-4]
        if method == "gt":
            continue

        gt_pose_file_path = experiment_dir/ f"{exp}/{method}/pose_file.txt"
        pose_dict = {}
        with open(gt_pose_file_path) as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                if line[0] == '#' or line == '':
                    continue
                data = line.split()
                tf_name= data[0]
                pq = np.array([float(val) for val in data[1:]])
                pose_dict[tf_name] = pq

        result = localization_results[image]["PnP_ret"]
        
        if result["success"]:
            tvec = result['tvec']
            qvec = result['qvec']
            rot = pr.matrix_from_quaternion(qvec)
            T_cam_world = pt.transform_from(rot, tvec)
            T_cam_body = pt.transform_from_pq(pose_dict["tf_cam_body"])

            T_map_body = pt.invert_transform(T_cam_world) @ T_cam_body
            pq_map_body = pt.pq_from_transform(T_map_body)
            result_dict[exp][method] = pq_map_body.tolist()
        else:
            result_dict[exp][method] = np.array([0,0,0,1,0,0,0]).tolist()
    return result_dict



def correct_names(directory):
    #root, dirs, files = os.walk(directory)
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            for parent, subdirs, files in os.walk(os.path.join(directory, dir)):
                parent_dir_name = os.path.basename(root)
                new_filename = f"{parent_dir_name}_{dir}.jpg"
                old_path = os.path.join(parent, "loc_image.jpg")
                new_path = os.path.join(parent, new_filename)
                try:
                    os.rename(old_path, new_path)
                except Exception as e:
                    print(e)
                #print(f"Renamed '{file}' to '{new_filename}' in '{root}'")

def copy_jpg_files(source_dir, destination_dir):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Traverse the source directory and its subdirectories
    for root, _, files in os.walk(source_dir):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                source_path = os.path.join(root, filename)
                if "IMG" not in source_path:
                    destination_path = os.path.join(destination_dir, filename)
                    shutil.copy2(source_path, destination_path)  # Use copy2 to preserve metadata




def main():
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DLAB_6")
    args = parser.parse_args()
    home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")
    environment = args.environment
    dataset = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    images = dataset

    # experiment_dir = Path("/local/home/hanlonm/data/Spot_Experiment/230823/23-08-23-09-01-02")
    experiment_dir = Path("/local/home/hanlonm/data/Spot_Experiment/230825")
    experiments = os.listdir(str(experiment_dir))

    result_dict = {}

    for experiment in experiments:
        print(experiment)
        # correct_names(str(experiment_dir/experiment))
        # copy_jpg_files(experiment_dir/experiment, images/"spot_eval")
        
        #if "40-32" in experiment:
        result_dict = gt_body_loc(result_dict, dataset=dataset, experiment_dir=experiment_dir/experiment, home_dir=home_dir, environment=environment, images=images)
    
    
    result_dict = method_body_loc(result_dict, dataset=dataset, experiment_dir=experiment_dir, home_dir=home_dir, environment=environment, images=images)
    print()

    # Save the dictionary to a JSON file
    with open(experiment_dir/'result_dict.json', 'w') as f:
        json.dump(result_dict, f)
    


if __name__ == "__main__":
    
    main()
