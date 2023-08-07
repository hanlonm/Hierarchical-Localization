import pycolmap
from hloc.utils import viz_3d
import matplotlib.pyplot as plt
from hloc.utils.read_write_model import Image
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


def get_min_max_viewing_distances(
        landmark_idx: int, pose_dict: dict,
        reconstruction: pycolmap.Reconstruction) -> np.ndarray:
    distances, num_observers = landmark_distances_and_observers(landmark_idx, pose_dict, reconstruction)
    return np.array([np.min(distances), np.max(distances)]), num_observers



# Plotting results
fig_matplot = plt.figure()
ax = fig_matplot.add_subplot(projection='3d')
#ax = view_transform_manager.plot_frames_in("map", s=1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.axes.set_xlim3d(left=-20, right=20)
ax.axes.set_ylim3d(bottom=-20, top=20)
ax.axes.set_zlim3d(bottom=-4, top=4)
ax.set_aspect('equal')
ax.grid(False)


T_cam_base = pt.transform_from(
            np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]),
            np.array([0, 0.0, 0.0]))

fig = viz_3d.init_figure()

output_folder = "DLAB_3"
#reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_reference_empty")
reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/{}/reconstruction".format(output_folder))


for image_id, image in reconstruction.images.items():
    image: Image
    # print(image_id, image)
    T_cam_world = pt.transform_from(pr.matrix_from_quaternion(image.qvec), image.tvec)
    T_world_base = np.linalg.inv(T_cam_world) @ T_cam_base
    pq = pt.pq_from_transform(T_world_base)
    ax.scatter3D(pq[0],pq[1],pq[2], s=1)
    # print([pq[0],pq[1],pq[2]])
    #viz_3d.plot_camera_colmap(fig, image, reconstruction.cameras[0])

for point3D_id, point3D in reconstruction.points3D.items():
    track = point3D.track
    image_ids = [element.image_id for element in point3D.track.elements]
    images = dict((key, reconstruction.images[key]) for key in image_ids)

    # print(image_ids)
    # print(track.elements)
    # print(point3D_id, point3D)

# for camera_id, camera in reconstruction.cameras.items():
#     print(camera_id, camera)
print(reconstruction.summary())
mre = reconstruction.compute_mean_reprojection_error()
print(reconstruction.filter_all_points3D(4, 0.1))
print(reconstruction.filter_observations_with_negative_depth())

landmark_list = list(reconstruction.points3D.keys())
pose_dict = pose_dict_from_reconstruction(reconstruction)
print("Calculating viewing angles and distances:")
min_obs_distance = 0.1
max_min_obs_distance = 8.0
min_num_observers = 5
count_1 = 0
count_2 = 0
viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping",cameras=False, )
fig.show()

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
    
print(count_1)
print(count_2)


fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.5)', name="mapping",cameras=False, )
#reconstruction.export_PLY('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_superpoint+superglue/00195.ply')
fig.show()
print(reconstruction.summary())
# reconstruction.write("/local/home/hanlonm/Hierarchical-Localization/outputs/{}/reconstruction".format(output_folder))
# plt.show()
