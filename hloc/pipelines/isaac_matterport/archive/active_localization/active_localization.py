from CMap2D import CMap2D
import numpy as np

import matplotlib.pyplot as plt
#from map2d import gridshow

import open3d as o3d
from scipy.spatial.transform import Rotation
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

from planning.rrt import RRTStar, path_smoothing
from planning.sampling import get_orientation_samples
from viewpoint import Camera, Viewpoint
from visibility_check import min_max_viewing_angle

def main():


    pcd = o3d.io.read_point_cloud("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_superpoint+superglue/00195.ply")
    pcd_points = np.asarray(pcd.points)




    # Plotting results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    #ax = view_transform_manager.plot_frames_in("map", s=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axes.set_xlim3d(left=-20, right=20)
    ax.axes.set_ylim3d(bottom=-20, top=20)
    ax.axes.set_zlim3d(bottom=-4, top=4)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.scatter3D(pcd_points[:, 0], pcd_points[:, 1], pcd_points[:, 2], s=1.0)

    plt.show()

if __name__ == "__main__":
    main()