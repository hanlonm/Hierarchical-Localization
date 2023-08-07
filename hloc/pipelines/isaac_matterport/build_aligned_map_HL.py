import os
import argparse
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
import pycolmap
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from tqdm import tqdm

def pose_dict_from_reconstruction(
        reconstruction: pycolmap.Reconstruction) -> dict:
    T_cam_base = pt.transform_from(
        np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
        np.array([0, 0.0, 0.0]))
    pose_dict = {}
    for image_id, image in reconstruction.images.items():
        T_cam_world = pt.transform_from(pr.matrix_from_quaternion(image.qvec),
                                        image.tvec)
        T_world_base = np.linalg.inv(T_cam_world) @ T_cam_base
        pq = pt.pq_from_transform(T_world_base)
        reorder = [0, 1, 2, 4, 5, 6, 3]
        pose_dict[image.name] = pq[reorder]
    return pose_dict

def landmark_distances_and_observers(landmark_idx: int, pose_dict: dict,
                       reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    point3D = reconstruction.points3D[landmark_idx]
    image_ids = [element.image_id for element in point3D.track.elements]
    image_ids = list(set(image_ids))
    num_observers = len(image_ids)

    images = dict((key, reconstruction.images[key]) for key in image_ids)
    frame_names = [image.name for image in images.values()]

    poses = dict((key, pose_dict[key]) for key in frame_names)
    vertices_that_see_landmark = np.vstack(list(poses.values()))

    connections = np.vstack([np.array(point3D.xyz)] *
                            len(image_ids)) - vertices_that_see_landmark[:, :3]

    distances = np.sqrt((connections * connections).sum(axis=1))

    return distances, num_observers

def optimize_reconstruction(reconstruction: pycolmap.Reconstruction, 
                            min_obs_distance=0.1, 
                            max_min_obs_distance=8.0, 
                            min_num_observers=3,
                            max_reprojection_error = 4) -> pycolmap.Reconstruction:
    reconstruction.filter_all_points3D(max_reprojection_error, 0.1)
    reconstruction.filter_observations_with_negative_depth()

    landmark_list = list(reconstruction.points3D.keys())
    pose_dict = pose_dict_from_reconstruction(reconstruction)
    count_1 = 0
    count_2 = 0

    for idx, landmark in enumerate(tqdm(landmark_list)):
        min_max_viewing_distances, num_observers = get_min_max_viewing_distances(
            landmark_idx=landmark,
            pose_dict=pose_dict,
            reconstruction=reconstruction)
        if min_max_viewing_distances[0] < min_obs_distance or min_max_viewing_distances[0] > max_min_obs_distance:
            reconstruction.delete_point3D(landmark)
            count_1 += 1
            continue
        if num_observers < min_num_observers:
            reconstruction.delete_point3D(landmark)
            count_2 += 1
    print(f"{count_1} points removed by observation distance")
    print(f"{count_2} points removed by num_observers")

    return reconstruction


def get_min_max_viewing_distances(
        landmark_idx: int, pose_dict: dict,
        reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    distances, num_observers = landmark_distances_and_observers(landmark_idx, pose_dict, reconstruction)
    return np.array([np.min(distances), np.max(distances)]), num_observers

def build_map(environment, align, optimize):
    home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")
    images = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    outputs = Path(home_dir+ '/Hierarchical-Localization/outputs/'+environment+'/')
    sfm_pairs = outputs / 'pairs-netvlad.txt'
    sfm_dir = outputs / 'reconstruction'


    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['NN-superpoint']

    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)


    feature_path = extract_features.main(feature_conf, images, outputs)
    matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, matches, verbose=True, camera_mode=pycolmap.CameraMode.SINGLE)

    if align:
        image_names = []
        locations = []

        pose_file = images / "image_poses.txt"
        with open(pose_file) as f:
                for line in f.readlines():
                    line = line.rstrip('\n')
                    if line[0] == '#' or line == '':
                        continue
                    data = line.replace(',', ' ').split()
                    image_names.append(data[0])
                    locations.append([float(data[1]),float(data[2]),float(data[3])])

        similarity_transform = model.align_robust(image_names, locations, 3, max_error=0.05, min_inlier_ratio=0.3)
        model.write(sfm_dir)

    if optimize:
        model = optimize_reconstruction(model, min_obs_distance=0.1, max_min_obs_distance=8.0, min_num_observers=4, max_reprojection_error=4.0)
        model.write(sfm_dir)
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
    fig.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DLAB_3")
    parser.add_argument("--align", type=bool, default=True)
    parser.add_argument("--optimize", type=bool, default=True)
    
    args = parser.parse_args()
    build_map(args.environment, args.align, args.optimize)


if __name__ == "__main__":
    main()
