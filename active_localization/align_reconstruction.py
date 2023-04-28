import pycolmap
from hloc.utils import viz_3d
import matplotlib.pyplot as plt
from hloc.utils.read_write_model import Image
import numpy as np

output_folder = "00195_hl_no_poses"
#reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_test/sfm_reference_empty")
reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/{}/sfm_superpoint+superglue".format(output_folder))

gt_reconstruction = pycolmap.Reconstruction("/local/home/hanlonm/Hierarchical-Localization/outputs/00195_loc/sfm_superpoint+superglue")

fig = viz_3d.init_figure()

image_names = []
locations = []

pose_file = "/local/home/hanlonm/.local/share/ov/pkg/isaac_sim-2022.2.0/_output_headless/image_poses.txt"
with open(pose_file) as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            if line[0] == '#' or line == '':
                continue
            data = line.replace(',', ' ').split()
            image_names.append(data[0])
            locations.append([float(data[1]),float(data[2]),float(data[3])-1.8])


res = reconstruction.align_robust(image_names, locations, 3, max_error=12.0, min_inlier_ratio=0.1)
print(res)

# viz_3d.plot_reconstruction(fig, gt_reconstruction, color='rgba(0,255,0,0.2)', name="gt")
# viz_3d.plot_reconstruction(fig, reconstruction, color='rgba(255,0,0,0.2)', name="aligned")
# fig.show()

reconstruction.write("/local/home/hanlonm/Hierarchical-Localization/outputs/{}/sfm_superpoint+superglue".format(output_folder))
