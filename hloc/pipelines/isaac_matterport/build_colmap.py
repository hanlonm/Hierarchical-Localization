from pathlib import Path
import argparse

from hloc import extract_features, match_features, pairs_from_poses, triangulation, pairs_from_retrieval
from hloc.pipelines.isaac_matterport.utils import build_empty_colmap_model

environment = "00195_NN"
images = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+"00195")
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment)
dataset_dir = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/'+"00195")

ref_sfm_empty = outputs / 'sfm_reference_empty'
ref_sfm = outputs / 'sfm_superpoint+NN'

num_ref_pairs = 20
ref_pairs = outputs / f'pairs-db-dist{num_ref_pairs}.txt'
#sfm_pairs = outputs / 'pairs-netvlad.txt'


fconf = extract_features.confs['superpoint_max']
mconf = match_features.confs['NN-superpoint']
#retrieval_conf = extract_features.confs['netvlad']

#retrieval_path = extract_features.main(retrieval_conf, images, outputs)



# Build an empty COLMAP model containing only camera and images
# from the provided poses and intrinsics.
build_empty_colmap_model(dataset_dir, ref_sfm_empty)

#pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=5)
# Match reference images that are spatially close.
pairs_from_poses.main(ref_sfm_empty, ref_pairs, num_ref_pairs)

image_list = [p.relative_to(images).as_posix() for p in (images / 'mapping/').iterdir()]

# Extract, match, and triangulate the reference SfM model.
ffile = extract_features.main(conf=fconf,image_dir=images,export_dir=outputs)
mfile = match_features.main(mconf, ref_pairs, fconf['output'], outputs)
#mfile = match_features.main(mconf, sfm_pairs, fconf['output'], outputs)
triangulation.main(ref_sfm, ref_sfm_empty, images, ref_pairs, ffile, mfile, verbose=True)