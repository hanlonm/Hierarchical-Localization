import pickle


with open('/local/home/hanlonm/Hierarchical-Localization/outputs/00195_loc/00195_hloc_superpoint+superglue_netvlad20.txt_logs.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)