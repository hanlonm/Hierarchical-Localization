from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

environment = "00195_HL_SIFT"
images = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'/')
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'reconstruction'


retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['sift']
matcher_conf = match_features.confs['NN-superpoint']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)


feature_path = extract_features.main(feature_conf, images, outputs)
matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, matches, verbose=True)

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

similarity_transform = model.align_robust(image_names, locations, 3, max_error=12.0, min_inlier_ratio=0.1)
model.write(sfm_dir)

fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
fig.show()