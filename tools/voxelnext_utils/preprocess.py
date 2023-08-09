
import numpy as np
from pcdet.utils import common_utils

tv = None
try:
  import cumm.tensorview as tv
except:
  pass


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


class VoxelGeneratorWrapper:
  def __init__(
          self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel,
          max_num_voxels):
    try:
      from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
      self.spconv_ver = 1
    except:
      try:
        from spconv.utils import VoxelGenerator
        self.spconv_ver = 1
      except:
        from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
        self.spconv_ver = 2

    if self.spconv_ver == 1:
      self._voxel_generator = VoxelGenerator(
          voxel_size=vsize_xyz,
          point_cloud_range=coors_range_xyz,
          max_num_points=max_num_points_per_voxel,
          max_voxels=max_num_voxels
      )
    else:
      self._voxel_generator = VoxelGenerator(
          vsize_xyz=vsize_xyz,
          coors_range_xyz=coors_range_xyz,
          num_point_features=num_point_features,
          max_num_points_per_voxel=max_num_points_per_voxel,
          max_num_voxels=max_num_voxels
      )

  def generate(self, points):
    if self.spconv_ver == 1:
      voxel_output = self._voxel_generator.generate(points)
      if isinstance(voxel_output, dict):
        voxels, coordinates, num_points = \
            voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
      else:
        voxels, coordinates, num_points = voxel_output
    else:
      assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
      voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
      tv_voxels, tv_coordinates, tv_num_points = voxel_output
      # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
      voxels = tv_voxels.numpy()
      coordinates = tv_coordinates.numpy()
      num_points = tv_num_points.numpy()
    return voxels, coordinates, num_points


class DataProcessor:
  def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
    self.point_cloud_range = point_cloud_range
    self.training = training
    self.num_point_features = num_point_features
    self.mode = 'train' if training else 'test'
    self.grid_size = self.voxel_size = None
    self.data_processor_queue = []
    self.data_processor_cfg = processor_configs
    self.voxel_generator = None

  def mask_points_and_boxes_outside_range(self, points):
    mask = common_utils.mask_points_by_range(points, self.point_cloud_range)
    return points[mask]

  def shuffle_points(self, points, config=None):
    if config.SHUFFLE_ENABLED[self.mode]:
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

  def transform_points_to_voxels(self, points, use_lead_xyz=False, config=None):
    if self.voxel_size is None:
      self.voxel_size = config.VOXEL_SIZE

    if self.grid_size is None:
      grid_size = (
        self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
      self.grid_size = np.round(grid_size).astype(np.int64)

    if self.voxel_generator is None:
      self.voxel_generator = VoxelGeneratorWrapper(
          vsize_xyz=self.voxel_size,
          coors_range_xyz=self.point_cloud_range,
          num_point_features=self.num_point_features,
          max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
          max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
      )

    voxel_output = self.voxel_generator.generate(points)
    voxels, coordinates, num_points = voxel_output

    if not use_lead_xyz:
      voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

    if config.get('DOUBLE_FLIP', False):
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

  def sample_points(self, points, config=None):
    num_points = config.NUM_POINTS[self.mode]
    if num_points == -1:
      return points

    if num_points < len(points):
      pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
      pts_near_flag = pts_depth < 40.0
      far_idxs_choice = np.where(pts_near_flag == 0)[0]
      near_idxs = np.where(pts_near_flag == 1)[0]
      choice = []
      if num_points > len(far_idxs_choice):
        near_idxs_choice = np.random.choice(
            near_idxs, num_points - len(far_idxs_choice),
            replace=False)
        choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
            if len(far_idxs_choice) > 0 else near_idxs_choice
      else:
        choice = np.arange(0, len(points), dtype=np.int32)
        choice = np.random.choice(choice, num_points, replace=False)
      np.random.shuffle(choice)
    else:
      choice = np.arange(0, len(points), dtype=np.int32)
      if num_points > len(points):
        extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
        choice = np.concatenate((choice, extra_choice), axis=0)
      np.random.shuffle(choice)
    return points[choice]

  def preprocess(self, points, use_lead_xyz):
    for dp_cfg in self.data_processor_cfg:
      if dp_cfg.NAME == 'mask_points_and_boxes_outside_range':
        points = self.mask_points_and_boxes_outside_range(points)
      elif dp_cfg.NAME == 'shuffle_points':
        points = self.shuffle_points(points, config=dp_cfg)
      elif dp_cfg.NAME == 'transform_points_to_voxels':
        voxels, coordinates, num_points = self.transform_points_to_voxels(
            points, use_lead_xyz, config=dp_cfg)
      else:
        raise ValueError('unimplemented processor function')

    return voxels, coordinates, num_points
