import numpy as np
from scipy.spatial.transform import Rotation as R
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


class Camera:

    def __init__(self, width, height, fx, fy, cx, cy) -> None:
        self.width = width
        self.height = height
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.fov_w = 2 * np.arctan(width / (2 * fx))
        self.fov_h = 2 * np.arctan(height / (2 * fy))


class Viewpoint:

    def __init__(self, camera: Camera, T_base_map, T_cam_base) -> None:
        self.camera = camera

        self.transform_manager = TransformManager()
        # T from map to camera
        self.T_base_map = T_base_map
        self.T_cam_base = T_cam_base
        self.transform_manager.add_transform("map", "base", T_base_map)
        self.transform_manager.add_transform("base", "cam", T_cam_base)

        self.T_cam_map = self.transform_manager.get_transform("map", "cam")

    def visible_points(self, point_cloud, clip_dist=None):
        point_cloud_cam = np.zeros((point_cloud.shape[0], 4))
        for i, point in enumerate(point_cloud):
            point_cloud_cam[i] = self.T_cam_map @ point

        # filter
        if clip_dist is not None:
            valid_z = np.logical_and(point_cloud_cam[:, 2] > 0,
                                     point_cloud_cam[:, 2] < clip_dist)
        else:
            valid_z = np.where(point_cloud_cam[:, 2] > 0)
        point_cloud_fov = point_cloud[valid_z]
        point_cloud_cam = point_cloud_cam[valid_z]

        projected_points = np.zeros((point_cloud_cam.shape[0], 3))

        # project points
        for i, point in enumerate(point_cloud_cam):
            px_point = self.camera.K @ point[:3]
            px_point /= px_point[2]
            projected_points[i] = px_point

        # filter points
        condition1 = np.logical_and(projected_points[:, 0] < self.camera.width,
                                    projected_points[:, 0] > 0)
        condition2 = np.logical_and(
            projected_points[:, 1] < self.camera.height,
            projected_points[:, 1] > 0)
        condition = np.logical_and(condition1, condition2)
        point_cloud_fov = point_cloud_fov[condition]
        projected_points = projected_points[condition]
        point_cloud_cam = point_cloud_cam[condition]

        return point_cloud_fov

    def visible_points_idx(self, point_cloud, clip_dist=None):
        point_cloud_cam = np.zeros((point_cloud.shape[0], 4))
        for i, point in enumerate(point_cloud):
            point_cloud_cam[i] = self.T_cam_map @ point
        # filter
        if clip_dist is not None:
            valid_z_idx = np.logical_and(point_cloud_cam[:, 2] > 0,
                                         point_cloud_cam[:, 2] < clip_dist)
        else:
            valid_z_idx = np.where(point_cloud_cam[:, 2] > 0)

        projected_points = np.zeros((point_cloud_cam.shape[0], 3))
        # project points
        for i, point in enumerate(point_cloud_cam):
            px_point = self.camera.K @ point[:3]
            px_point /= px_point[2]
            projected_points[i] = px_point

        # filter points
        condition1 = np.logical_and(projected_points[:, 0] < self.camera.width,
                                    projected_points[:, 0] > 0)
        condition2 = np.logical_and(
            projected_points[:, 1] < self.camera.height,
            projected_points[:, 1] > 0)
        fov_condition = np.logical_and(condition1, condition2)

        valid_idx = np.logical_and(fov_condition, valid_z_idx)

        return valid_idx

    def visible_points_viewing_angles(self,
                                      point_cloud: np.ndarray,
                                      min_max_viewing_angles: np.ndarray,
                                      clip_dist: int = None) -> np.ndarray:
        point_cloud_cam = np.zeros((point_cloud.shape[0], 4))
        for i, point in enumerate(point_cloud):
            point_cloud_cam[i] = self.T_cam_map @ point

        # filter
        if clip_dist is not None:
            valid_z = np.logical_and(point_cloud_cam[:, 2] > 0,
                                     point_cloud_cam[:, 2] < clip_dist)
        else:
            valid_z = point_cloud_cam[:, 2] > 0
        point_cloud_visible = point_cloud[valid_z]
        point_cloud_cam = point_cloud_cam[valid_z]
        min_max_viewing_angles = min_max_viewing_angles[valid_z]

        projected_points = np.zeros((point_cloud_cam.shape[0], 3))

        # project points
        for i, point in enumerate(point_cloud_cam):
            px_point = self.camera.K @ point[:3]
            px_point /= px_point[2]
            projected_points[i] = px_point

        # filter points
        fov_condition_1 = np.logical_and(
            projected_points[:, 0] < self.camera.width,
            projected_points[:, 0] > 0)
        fov_condition_2 = np.logical_and(
            projected_points[:, 1] < self.camera.height,
            projected_points[:, 1] > 0)
        fov_condition = np.logical_and(fov_condition_1, fov_condition_2)
        point_cloud_visible = point_cloud_visible[fov_condition]
        projected_points = projected_points[fov_condition]
        point_cloud_cam = point_cloud_cam[fov_condition]

        min_max_viewing_angles = min_max_viewing_angles[fov_condition]

        # angles = np.arctan2(
        # connections[:, 1] * bearings[:, 0] -
        # connections[:, 0] * bearings[:, 1],
        # connections[:, 0] * bearings[:, 0] +
        # connections[:, 1] * bearings[:, 1])

        # Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
        pq_base_map = pt.pq_from_transform(np.linalg.inv(self.T_cam_map))
        rotation_mat = pr.matrix_from_quaternion(pq_base_map[3:])
        rotation = R.from_matrix(rotation_mat)

        connections = rotation.apply(point_cloud_cam[:, :3])
        bearings = bearings = np.vstack(
            [np.array([1, 0, 0]) * point_cloud_cam.shape[0]])
        viewing_angles = np.arctan2(
            connections[:, 1] * bearings[:, 0] -
            connections[:, 0] * bearings[:, 1],
            connections[:, 0] * bearings[:, 0] +
            connections[:, 1] * bearings[:, 1])

        # viewing_angles = np.arctan2(-1 * (point_cloud_cam[:, 0]),
        #                             point_cloud_cam[:, 2])

        # bearing = np.array([0, 1])
        # viewing_angles = np.arctan2(
        #     point_cloud_visible[:, 2] * bearing[:, 0] -
        #     point_cloud_visible[:, 0] * bearing[:, 1],
        #     point_cloud_visible[:, 0] * bearing[:, 0] +
        #     point_cloud_visible[:, 1] * bearing[:, 1])

        angle_condition_1 = viewing_angles > min_max_viewing_angles[:, 0]
        angle_condition_2 = viewing_angles < min_max_viewing_angles[:, 1]
        angle_condition = np.logical_and(angle_condition_1, angle_condition_2)

        point_cloud_visible = point_cloud_visible[angle_condition]

        return point_cloud_visible
