import os
import shutil
import argparse
from pathlib import Path
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
from pytransform3d.transformations import random_transform
from pytransform3d.transform_manager import TransformManager

from skspatial.objects import Plane, Points
from skspatial.plotting import plot_3d

from scipy.spatial.transform import Rotation as R



def get_rotation_matrix(vec2, vec1=np.array([1, 0, 0])):
    """get rotation matrix between two vectors using scipy"""
    vec1 = np.reshape(vec1, (1, -1))
    vec2 = np.reshape(vec2, (1, -1))
    r = R.align_vectors(vec2, vec1)
    return r[0].as_matrix()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="DLAB_3")
    
    args = parser.parse_args()
    environment = args.environment
    home_dir = os.environ.get("CLUSTER_HOME", "/local/home/hanlonm")
    dataset_dir = Path(home_dir + '/Hierarchical-Localization/datasets/'+environment)
    trajectories_file = np.loadtxt(dataset_dir / "trajectories.txt", dtype=str, delimiter=",")

    tm = TransformManager()

    T_rig_cam = pt.transform_from_pq(np.array([0.0,0.0,0.0,-0.7068309371386561, -0.00040853092064318874, -0.00020564327201549008, 0.7073823698092996]))
    T_cam_base = pt.transform_from(
                np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
                np.array([0, 0.0, 0.0]))

    trajectories_file = trajectories_file[::50]
    point_list = []
    for i, pose_line in enumerate(trajectories_file[:,:]):
        ts = pose_line[0]
        qp_wxyz_xyz = pose_line[2:].astype(float)
        pq = qp_wxyz_xyz[[4, 5, 6, 0, 1, 2, 3]]
        point_list.append(pq[:3])


    points = Points(point_list)

    plane = Plane.best_fit(points)
    print(plane.normal)

    #align_matrix = align_to_xy_plane(normal=plane.normal)

    # align_rot_matrix = pr.matrix_from_two_vectors( plane.normal, np.array([0,1,0]))

    align_rot_matrix = get_rotation_matrix(-1*plane.normal, np.array([0, 0, 1]))
    print("rotation:", f"plane normal is {plane.normal}", f"\nalign_rot_matrix @ plane.normal is {align_rot_matrix.T @ plane.normal}")
    align_rot_matrix = pr.check_matrix(align_rot_matrix)



    T_rotworld_world = pt.transform_from(align_rot_matrix.T, np.array([0,0,0]))
    plot_3d(
        points.plotter(c='k', s=50, depthshade=False),
        plane.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5)),
    )
    plt.show()

    for i, pose_line in enumerate(trajectories_file[:,:]):
        ts = pose_line[0]
        qp_wxyz_xyz = pose_line[2:].astype(float)
        pq = qp_wxyz_xyz[[4, 5, 6, 0, 1, 2, 3]]
        T_world_rig = pt.transform_from_pq(pq)
        T_world_cam = T_world_rig @ T_rig_cam
        T_world_base = T_world_cam @ T_cam_base

        T_worldrot_base = T_rotworld_world @ T_world_base

        
        
        tm.add_transform(f"base_rot_{i}", "world", T_worldrot_base)
        #tm.add_transform(f"base_{i}", "world", T_world_base)

    tm.plot_frames_in("world", show_name=True)
    # tm.plot_frames_in("world", whitelist=["world"])

    # print(tm.get_transform("cam_al_50", "cam_al_25"))

    plt.show()



if __name__ == "__main__":
    main()