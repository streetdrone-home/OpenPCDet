
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
from voxelnext_utils.preprocess import PointFeatureEncoder, DataProcessor
from voxelnext_utils.file_io import get_lidar_with_sweeps

tv = None
try:
  import cumm.tensorview as tv
except:
  pass


root_path = Path("/home/cuda_pp/OpenPCDet/data/nuscenes/v1.0-mini")

info_path = Path("/home/cuda_pp/OpenPCDet/data/nuscenes/v1.0-mini/nuscenes_infos_10sweeps_val.pkl")

with open(info_path, 'rb') as f:
  infos = pickle.load(f)
  info = infos[3]
  # pprint(infos[1])


cfg_file = "/home/cuda_pp/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"

cfg_from_yaml_file(cfg_file, cfg)
cfg.TAG = Path(cfg_file).stem
cfg.EXP_GROUP_PATH = '/'.join(cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

dataset_cfg = cfg.DATA_CONFIG

batch_size = 1

print("############# DATASET CONFIG ###################\n", dataset_cfg)


points = get_lidar_with_sweeps(root_path, info, max_sweeps=dataset_cfg.MAX_SWEEPS)

point_feature_encoder = PointFeatureEncoder(
    dataset_cfg.POINT_FEATURE_ENCODING,
    point_cloud_range=dataset_cfg.POINT_CLOUD_RANGE
)
num_point_features = point_feature_encoder.num_point_features
print("points shape: ", points.shape)
points, use_lead_xyz = point_feature_encoder.forward(points)
print("points shape after encoding: ", points.shape)

point_cloud_range = np.array(dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
data_p = DataProcessor(dataset_cfg.DATA_PROCESSOR, point_cloud_range, False, num_point_features)
voxels, coordinates, num_points = data_p.preprocess(points, use_lead_xyz)

# pad coordinates for single inference
coordinates = np.pad(coordinates, ((0, 0), (1, 0)), mode='constant', constant_values=0)

voxels = torch.from_numpy(voxels).float().cuda()
coordinates = torch.from_numpy(coordinates).float().cuda()
num_points = torch.from_numpy(num_points).float().cuda()

print('v size: ', voxels.size())
print('v num pts: ', num_points.size())
print('v coords: ', coordinates.size())

# For inference time
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model = VoxelNext(cfg.MODEL, cfg.CLASS_NAMES, num_point_features,
                  data_p.grid_size, point_cloud_range, data_p.voxel_size)


model_ckpt = "/home/cuda_pp/OpenPCDet/ckpts/voxelnext_nuscenes_kernel1.pth"
log_file = ('log_test_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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


out_dict = {}
for key, value in pred_dicts[0].items():
  print(key)
  print(type(value))
  if not isinstance(value, list):
    out_dict[key] = value.cpu()

# torch.save(out_dict, 'pred_dicts1.pt')
