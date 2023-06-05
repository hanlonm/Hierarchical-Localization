from pathlib import Path

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_exhaustive, pairs_from_retrieval
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

environment = "00195_hl_no_poses"
images = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'/')
# sfm_pairs = outputs / 'pairs-sfm.txt'
# loc_pairs = outputs / 'pairs-loc.txt'
# sfm_dir = outputs / 'sfm'
# features = outputs / 'features.h5'
# matches = outputs / 'matches.h5'
sfm_pairs = outputs / 'pairs-netvlad.txt'
sfm_dir = outputs / 'sfm_superpoint+superglue'


retrieval_conf = extract_features.confs['netvlad']
feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']

retrieval_path = extract_features.main(retrieval_conf, images, outputs)
pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)



#references = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]
#references = references[0::10]
# print(len(references), "mapping images")
# plot_images([read_image(images / r) for r in references[:4]], dpi=50)

# extract_features.main(feature_conf, images, image_list=references, feature_path=features)
# pairs_from_exhaustive.main(sfm_pairs, image_list=references)
# match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

feature_path = extract_features.main(feature_conf, images, outputs)
matches = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, matches, verbose=True)
#model.export_PLY(str(outputs / environment) + '.ply')

fig = viz_3d.init_figure()
viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
fig.show()

#visualization.visualize_sfm_2d(model, images, color_by='visibility', n=2)   