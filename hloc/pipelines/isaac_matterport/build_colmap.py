from pathlib import Path
import argparse

from ... import extract_features, match_features
from ... import pairs_from_poses, triangulation
from .utils import get_timestamps, delete_unused_images
from .utils import build_empty_colmap_model

environment = "00195"
images = Path('/local/home/hanlonm/git/trajectory_eval/output/images')
outputs = Path('/local/home/hanlonm/Hierarchical-Localization/outputs/'+environment+'/')
ref_dir = Path('/local/home/hanlonm/Hierarchical-Localization/datasets/00195')

ref_sfm_empty = outputs / 'sfm_reference_empty'
ref_sfm = outputs / 'sfm_superpoint+superglue'

num_ref_pairs = 20
ref_pairs = outputs / f'pairs-db-dist{num_ref_pairs}.txt'

fconf = extract_features.confs['superpoint_max']
mconf = match_features.confs['superglue']

# Build an empty COLMAP model containing only camera and images
# from the provided poses and intrinsics.
build_empty_colmap_model(ref_dir, ref_sfm_empty)