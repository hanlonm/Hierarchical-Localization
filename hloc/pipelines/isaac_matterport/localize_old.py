from pathlib import Path
from hloc import extract_features, match_features, pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm, visualization

environment = "00195"
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'/')
loc_outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'_loc/')
dataset = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+environment)
build_images = dataset / 'mapping'
localization_images = dataset / 'localization'
loc_pairs = outputs / 'pairs-query-netvlad20.txt'  # top 20 retrieved by NetVLAD

local_features = outputs / 'feats-superpoint-n4096-rmax1600_loc.h5'
global_features = outputs / 'global-feats-netvlad_loc.h5'

references = [p.relative_to(dataset).as_posix() for p in (dataset / 'mapping/').iterdir()]

feature_conf = extract_features.confs['superpoint_max']
matcher_conf = match_features.confs['superglue']
retrieval_conf = extract_features.confs['netvlad']

retrieval_path = extract_features.main(retrieval_conf, build_images, outputs)
#pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)


matches = outputs / 'loc_matches.h5'
query = 'localization/frame_162.jpeg'

extract_features.main(feature_conf, dataset, image_list=[query], feature_path=local_features, overwrite=True)
retrieval_path=extract_features.main(retrieval_conf, dataset, image_list=[query]+references, feature_path=global_features,overwrite=False)
pairs_from_retrieval.main(retrieval_path, loc_pairs, num_matched=1)
match_features.main(matcher_conf, loc_pairs, features=local_features, matches=matches, overwrite=True)


