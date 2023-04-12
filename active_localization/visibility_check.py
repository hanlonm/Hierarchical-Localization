import numpy as np
from scipy.spatial.transform import Rotation as R


def viewing_angles(landmark_idx: int, landmarks: np.ndarray,
                   vertices: np.ndarray, observations: np.ndarray):
    observed = observations[observations[:, 3] == landmark_idx, 0]
    vertices_that_see_landmark = vertices[observed.astype(int), 2:9]

    # r = R.from_quat(vertices_that_see_landmark[:, 3:7])
    # bearings = r.apply(np.array([1, 0, 0]))

    bearings = np.vstack(
        [np.array([1, 0, 0]) * vertices_that_see_landmark.shape[0]])

    connections = np.vstack(
        [landmarks[landmark_idx, 1:4]] * vertices_that_see_landmark.shape[0]
    ) - vertices_that_see_landmark[:, :3]

    angles = np.arctan2(
        connections[:, 1] * bearings[:, 0] -
        connections[:, 0] * bearings[:, 1],
        connections[:, 0] * bearings[:, 0] +
        connections[:, 1] * bearings[:, 1])

    return angles


def min_max_viewing_angle(landmark_idx: int, landmarks: np.ndarray,
                          vertices: np.ndarray,
                          observations: np.ndarray) -> np.ndarray:
    angles = viewing_angles(landmark_idx, landmarks, vertices, observations)

    return np.array([np.min(angles), np.max(angles)])
