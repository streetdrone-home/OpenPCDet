
import numpy as np
from pcdet.utils import common_utils

from spconv.pytorch.utils import PointToVoxel
import torch


class DataProcessor:
  def __init__(
          self, processor_configs, point_cloud_range, training, num_point_features,
          device=torch.device('cuda:0')):
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

        self.voxel_generator = PointToVoxel(
          vsize_xyz=self.voxel_size,
          coors_range_xyz=self.point_cloud_range,
          num_point_features=self.num_point_features,
          max_num_voxels=max_num_voxels,
          max_num_points_per_voxel=max_pts_per_voxel,
          device=device
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

  def transform_points_to_voxels(self, points):
    voxels, coordinates, num_points = self.voxel_generator(points)

    if self.is_double_flip:
      voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
      points_yflip, points_xflip, points_xyflip = self.double_flip(points)
      points_list = [points_yflip, points_xflip, points_xyflip]
      keys = ['yflip', 'xflip', 'xyflip']
      for i, key in enumerate(keys):
        voxel_output = self.voxel_generator(points_list[i])
        voxels, coordinates, num_points = voxel_output
        voxels_list.append(voxels)
        voxel_coords_list.append(coordinates)
        voxel_num_points_list.append(num_points)

      return voxels_list, voxel_coords_list, voxel_num_points_list

    return voxels, coordinates, num_points

  def preprocess(self, points):
    return self.transform_points_to_voxels(
      self.shuffle_points(
        self.mask_points_and_boxes_outside_range(points)
      )
    )
