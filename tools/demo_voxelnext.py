
from functools import partial
from nw_models.voxelnext import VoxelNext
import pickle
from pathlib import Path
from pprint import pprint
import numpy as np
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils
import torch
import datetime
import time
import os
from voxelnext_utils.gpu_preprocess import DataProcessor
from voxelnext_utils.file_io import get_lidar_with_sweeps

from torch.nn import functional as torch_f

tv = None
try:
  import cumm.tensorview as tv
except:
  pass


root_path = Path("/home/cuda_pp/OpenPCDet/data/nuscenes/v1.0-mini")
info_path = Path("/home/cuda_pp/OpenPCDet/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl")
model_ckpt = "/home/cuda_pp/OpenPCDet/ckpts/voxelnext_nuscenes_kernel1.pth"
cfg_file = "/home/cuda_pp/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"

with open(info_path, 'rb') as f:
  infos = pickle.load(f)
  info = infos[21]

cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem
cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

dataset_cfg = cfg.DATA_CONFIG

batch_size = 1

# For inference time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("############# DATASET CONFIG ###################\n", dataset_cfg)

point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
num_point_features = len(dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)
data_p = DataProcessor(dataset_cfg.DATA_PROCESSOR, point_cloud_range, False, num_point_features)

points = get_lidar_with_sweeps(root_path, info, max_sweeps=dataset_cfg.MAX_SWEEPS)
np.save('points.npy', points)

start_time = time.time()
# no need of point feature encoding
points = torch.from_numpy(points).to(device=torch.device('cuda:0'))
voxels, coordinates, num_points = data_p.preprocess(points)
# pad coordinates for single inference
coordinates = torch_f.pad(coordinates, (1, 0), mode='constant', value=0)
end_time = time.time()
print("process time: ", end_time - start_time)

print('v size: ', voxels.size())
print('v num pts: ', num_points.size())
print('v coords: ', coordinates.size())


model = VoxelNext(cfg.MODEL, cfg.CLASS_NAMES, num_point_features,
                  data_p.grid_size, point_cloud_range, data_p.voxel_size)


log_file = ('logs/log_test_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
logger = common_utils.create_logger(log_file=log_file)
model.load_params_from_file(filename=model_ckpt, logger=logger)
model.cuda()
model.eval()

with torch.no_grad():
  start_time = time.time()
  pred_dicts, ret_dict = model(batch_size, voxels, num_points, coordinates)
  end_time = time.time()
  print('inference time: ', end_time - start_time)

pprint(pred_dicts)

torch.save(pred_dicts[0], 'pred_dicts1.pt')
