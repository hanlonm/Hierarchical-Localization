import h5py
import numpy as np

hf = h5py.File("/local/home/hanlonm/Hierarchical-Localization/outputs/00067/global-feats-netvlad.h5", "a")
print(hf.keys())
#del hf["localization"]
#print(hf['localization-1684495957_070_max_crit_wp_004.jpeg'].keys())

# fs = h5py.File('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_HL/feats-superpoint-n4096-rmax1600.h5', 'r')
# fd = h5py.File('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_HL/feats-superpoint-n4096-rmax1600_copy.h5', 'w')
# for a in fs.attrs:
#     fd.attrs[a] = fs.attrs[a]
# for d in fs:
#     if not 'SFS_TRANSITION' in d: fs.copy(d, fd)