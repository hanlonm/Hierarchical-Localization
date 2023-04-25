import pycolmap
from hloc.utils import viz_3d
import matplotlib.pyplot as plt
from hloc.utils.read_write_model import Image
import numpy as np


# Plotting results
# fig_matplot = plt.figure()
# ax = fig_matplot.add_subplot(projection='3d')
# #ax = view_transform_manager.plot_frames_in("map", s=1)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.axes.set_xlim3d(left=-20, right=20)
# ax.axes.set_ylim3d(bottom=-20, top=20)
# ax.axes.set_zlim3d(bottom=-4, top=4)
# ax.set_aspect('equal')
# ax.grid(False)


fig = viz_3d.init_figure()

#reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_reference_empty")
reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/warehouse/sfm_superpoint+superglue")
print(reconstruction.summary())

for image_id, image in reconstruction.images.items():
    image: Image
    print(image_id, image)
    #ax.scatter3D(image.tvec[0],image.tvec[1],image.tvec[2], s=1)
    #viz_3d.plot_camera_colmap(fig, image, reconstruction.cameras[0])

for point3D_id, point3D in reconstruction.points3D.items():
    track = point3D.track
    image_ids = [element.image_id for element in point3D.track.elements]
    images = dict((key, reconstruction.images[key]) for key in image_ids)

    # print(image_ids)
    # print(track.elements)
    # print(point3D_id, point3D)

for camera_id, camera in reconstruction.cameras.items():
    print(camera_id, camera)
    
viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(0,0,255,0.5)', name="mapping")
#reconstruction.export_PLY('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_superpoint+superglue/00195.ply')
fig.show()
#plt.show()
