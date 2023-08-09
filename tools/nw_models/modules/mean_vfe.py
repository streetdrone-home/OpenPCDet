import torch
import torch.nn as nn


class MeanVFE(nn.Module):
  def __init__(self, num_point_features):
    super().__init__()
    self.num_point_features = num_point_features

  def get_output_feature_dim(self):
    return self.num_point_features

  def forward(self, voxels, voxel_num_points):
    """
    Args:
        voxels: (num_voxels, max_points_per_voxel, C)
        voxel_num_points: optional (num_voxels)

    Returns:
        vfe_features: (num_voxels, C)
    """
    points_mean = voxels[:, :, :].sum(dim=1, keepdim=False)
    normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1.0).type_as(voxels)
    points_mean = points_mean / normalizer
    return points_mean.contiguous()
