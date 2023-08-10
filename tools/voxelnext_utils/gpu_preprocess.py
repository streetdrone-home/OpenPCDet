
import numpy as np
from pcdet.utils import common_utils

from spconv.pytorch.utils import PointToVoxel
import torch


class PointFeatureEncoder:
  def __init__(self, config, point_cloud_range=None):
    super().__init__()
    self.point_encoding_config = config
    assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
    self.used_feature_list = self.point_encoding_config.used_feature_list
    self.src_feature_list = self.point_encoding_config.src_feature_list
    self.point_cloud_range = point_cloud_range

  @property
  def num_point_features(self):
    return getattr(self, self.point_encoding_config.encoding_type)(points=None)

  def forward(self, points):
    """
    Args:
        data_dict:
            points: (N, 3 + C_in)
            ...
    Returns:
        data_dict:
            points: (N, 3 + C_out),
            use_lead_xyz: whether to use xyz as point-wise features
            ...
    """
    points, use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
        points
    )

    if self.point_encoding_config.get(
            'filter_sweeps', False) and 'timestamp' in self.src_feature_list:
      max_sweeps = self.point_encoding_config.max_sweeps
      idx = self.src_feature_list.index('timestamp')
      dt = np.round(points[:, idx], 2)
      max_dt = sorted(np.unique(dt))[min(len(np.unique(dt)) - 1, max_sweeps - 1)]
      points = points[dt <= max_dt]

    return points, use_lead_xyz

  def absolute_coordinates_encoding(self, points=None):
    """
    Args:
        points: (N, 3 + C_in)
    Returns:
        point_features,
        bool: use_lead_xyz
        ...
    """
    if points is None:
      num_output_features = len(self.used_feature_list)
      return num_output_features

    assert points.shape[-1] == len(self.src_feature_list)
    point_feature_list = [points[:, 0:3]]
    for x in self.used_feature_list:
      if x in ['x', 'y', 'z']:
        continue
      idx = self.src_feature_list.index(x)
      point_feature_list.append(points[:, idx:idx + 1])
    point_features = np.concatenate(point_feature_list, axis=1)

    return point_features, True


class VoxelGenerator:
  def __init__(
          self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel,
          max_num_voxels):
    self._voxel_generator = PointToVoxel(
      vsize_xyz=vsize_xyz,
      coors_range_xyz=coors_range_xyz,
      num_point_features=num_point_features,
      max_num_voxels=max_num_voxels,
      max_num_points_per_voxel=max_num_points_per_voxel,
      device=torch.device('cuda:0')
    )

  def generate(self, points):
    pts = torch.from_numpy(points).to(self._voxel_generator.device)
    return self._voxel_generator(pts)


class DataProcessor:
  def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
    self.point_cloud_range = point_cloud_range
    self.training = training
    self.num_point_features = num_point_features
    self.mode = 'train' if training else 'test'
    self.voxel_generator = None

    for dp_cfg in processor_configs:
      if dp_cfg.NAME == 'mask_points_and_boxes_outside_range':
        pass
      elif dp_cfg.NAME == 'shuffle_points':
        self.shuffle_enabled = dp_cfg.SHUFFLE_ENABLED[self.mode]
      elif dp_cfg.NAME == 'transform_points_to_voxels':
        self.voxel_size = dp_cfg.VOXEL_SIZE
        grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)
        max_pts_per_voxel = dp_cfg.MAX_POINTS_PER_VOXEL
        max_num_voxels = dp_cfg.MAX_NUMBER_OF_VOXELS[self.mode]
        self.is_double_flip = dp_cfg.get('DOUBLE_FLIP', False)

        self.voxel_generator = VoxelGenerator(
          vsize_xyz=self.voxel_size,
          coors_range_xyz=self.point_cloud_range,
          num_point_features=self.num_point_features,
          max_num_points_per_voxel=max_pts_per_voxel,
          max_num_voxels=max_num_voxels,
        )
      else:
        raise ValueError('unimplemented processor function')

  def mask_points_and_boxes_outside_range(self, points):
    mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
    return points[mask]

  def shuffle_points(self, points):
    if self.shuffle_enabled:
      shuffle_idx = np.random.permutation(points.shape[0])
      points = points[shuffle_idx]

    return points

  def double_flip(self, points):
    # y flip
    points_yflip = points.copy()
    points_yflip[:, 1] = -points_yflip[:, 1]

    # x flip
    points_xflip = points.copy()
    points_xflip[:, 0] = -points_xflip[:, 0]

    # x y flip
    points_xyflip = points.copy()
    points_xyflip[:, 0] = -points_xyflip[:, 0]
    points_xyflip[:, 1] = -points_xyflip[:, 1]

    return points_yflip, points_xflip, points_xyflip

  def transform_points_to_voxels(self, points, use_lead_xyz=False):
    voxels, coordinates, num_points = self.voxel_generator.generate(points)

    if not use_lead_xyz:
      voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

    if self.is_double_flip:
      voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
      points_yflip, points_xflip, points_xyflip = self.double_flip(points)
      points_list = [points_yflip, points_xflip, points_xyflip]
      keys = ['yflip', 'xflip', 'xyflip']
      for i, key in enumerate(keys):
        voxel_output = self.voxel_generator.generate(points_list[i])
        voxels, coordinates, num_points = voxel_output

        if not use_lead_xyz:
          voxels = voxels[..., 3:]
        voxels_list.append(voxels)
        voxel_coords_list.append(coordinates)
        voxel_num_points_list.append(num_points)

      return voxels_list, voxel_coords_list, voxel_num_points_list

    return voxels, coordinates, num_points

  def preprocess(self, points, use_lead_xyz):
    return self.transform_points_to_voxels(
      self.shuffle_points(
        self.mask_points_and_boxes_outside_range(points)
      ),
      use_lead_xyz
    )
