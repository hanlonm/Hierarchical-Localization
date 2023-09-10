import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager
from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt




def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()




def pre_process_hl_data(environment):
    home_dir = os.environ.get("BASE_DIR", "/local/home/hanlonm")
    dataset_dir = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    
    image_files = np.loadtxt(dataset_dir / "images.txt", dtype=str)
    trajectories_file = np.loadtxt(dataset_dir / "trajectories.txt", dtype=str, delimiter=",")
    rigs_file = np.loadtxt(dataset_dir / "rigs.txt", dtype=str)
    pose_file_name = dataset_dir / "image_poses.txt"
    pose_file = open(pose_file_name, "w")
    pose_file.write("# image_file tx ty tz qx qy qz qw \n")

    T_rig_camlf = pt.transform_from_pq(np.array([0.0,0.0,0.0,-0.7068309371386561, -0.00040853092064318874, -0.00020564327201549008, 0.7073823698092996]))
    T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                np.array([0, 0.0, 0.0]))
    
    
    # align plane to xy plane of coordinate system, pseudo gravity alignment
    point_list = []
    for i, pose_line in enumerate(trajectories_file[:,:]):
        ts = pose_line[0]
        qp_wxyz_xyz = pose_line[2:].astype(float)
        pq = qp_wxyz_xyz[[4, 5, 6, 0, 1, 2, 3]]
        point_list.append(pq[:3])
    points = Points(point_list)
    plane = Plane.best_fit(points)
    
    align_rot_matrix = get_rotation_matrix(-1*plane.normal, np.array([0, 0, 1]))
    print("rotation:", f"plane normal is {plane.normal}", f"\nalign_rot_matrix @ plane.normal is {align_rot_matrix.T @ plane.normal}")
    T_rotworld_world = pt.transform_from(align_rot_matrix.T, np.array([0,0,0]))
    plot_3d(
        points.plotter(c='k', s=50, depthshade=False),
        plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
    )
    plt.show()

    T_world_rig_dict = {}
    for pose_line in trajectories_file:
        ts = pose_line[0]
        qp_wxyz_xyz = pose_line[2:].astype(float)
        pq = qp_wxyz_xyz[[4, 5, 6, 0, 1, 2, 3]]
        T_world_rig = pt.transform_from_pq(pq)
        T_world_rig_dict[ts] = T_world_rig



    mapping_image_dir = dataset_dir / "mapping"
    mapping_image_dir.mkdir(exist_ok=True)

    tm = TransformManager()
    for i, image_file in enumerate(image_files):
        image_path = str(dataset_dir / "raw_data" / image_file[2])
        image_file_name = image_file[1][:-1] + "_" + image_file[0][:-1] + ".jpg"
        dst_path = str(mapping_image_dir / image_file_name)
        ts = image_file[0][:-1]
        # only use lf camera to align map
        if "lf" in image_file_name:
            T_world_rig = T_world_rig_dict[ts]
            T_world_cam = T_world_rig @ T_rig_camlf
            T_world_base = T_world_cam @ T_cam_base

            T_worldrot_base = T_rotworld_world @ T_world_base

            if i % 10 == 0:
                tm.add_transform(f"base_rot_{i}", "world", T_worldrot_base)



            pq = pt.pq_from_transform(T_worldrot_base)
            pose_file.write("mapping/" + image_file_name + " " +
                                str(pq[0]) + " " + str(pq[1]) +
                                " " + str(pq[2]) + " " +
                                str(pq[4]) + " " + str(pq[5]) +
                                " " + str(pq[6]) + " " +
                                str(pq[3]) + "\n")
        # shutil.copy(image_path, dst_path)
    
    pose_file.close()

    tm.plot_frames_in("world", show_name=False)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DLAB_3")
    
    args = parser.parse_args()
    pre_process_hl_data(args.environment)


if __name__ == "__main__":
    main()