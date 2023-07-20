import torch
import torch.nn as nn

from modules.mean_vfe import MeanVFE
from modules.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from modules.voxelnext_head import VoxelNeXtHead


class VoxelNext(nn.Module):
  def __init__(self, class_names, model_cfg) -> None:
    self.class_names = class_names
    self.num_class = len(self.class_names)
    self.register_buffer('global_step', torch.LongTensor(1).zero_())
    self.model_cfg = model_cfg

  @property
  def mode(self):
    return 'TRAIN' if self.training else 'TEST'

  def update_global_step(self):
    self.global_step += 1

  def build_network(self):
    model_info_dict = {
        'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
        'num_point_features': self.dataset.point_feature_encoder.num_point_features,
        'grid_size': self.dataset.grid_size,
        'point_cloud_range': self.dataset.point_cloud_range,
        'voxel_size': self.dataset.voxel_size,
        'depth_downsample_factor': self.dataset.depth_downsample_factor
    }
    # voxel feature encoder
    # forward
    # 	args: (voxels, voxel_num_points)
    # 	returns: vfe_features
    vfe = MeanVFE(model_info_dict['num_point_features'])
    model_info_dict['num_point_features'] = vfe.get_output_feature_dim()
    self.add_module("vfe", vfe)

    # backbone
    # has 4 more default args
    # spconv_kernel_sizes=[3,3,3,3],
    # channels=[16, 32, 64, 128, 128],
    # output_channels=128
    # forward
    # 	args: batch_size, vfe_features, voxel_coords
    # returns: {encoded_spconv_tensor: sparse tensor
    # encoded_spconv_tensor_stride: 8
    # multi_scale_3d_features:
    # 'x_conv1': x_conv1
    # 'x_conv2': x_conv2
    # 'x_conv3': x_conv3
    # 'x_conv4': x_conv4
    # multi_scale_3d_strides:
    # 'x_conv1': 1
    # 'x_conv2': 2
    # 'x_conv3': 4
    # 'x_conv4': 8
    # }
    backbone_3d = VoxelResBackBone8xVoxelNeXt(
        self.model_cfg, input_channels=model_info_dict['num_point_features'],
        grid_size=model_info_dict['grid_size'])
    model_info_dict['num_point_features'] = backbone_3d.num_point_features
    model_info_dict['backbone_channels'] = backbone_3d.backbone_channels \
        if hasattr(backbone_3d, 'backbone_channels') else None
    self.add_module("backbone3d", backbone_3d)

    # dense head
    dense_head = VoxelNeXtHead(
        self.model_cfg, num_class=self.num_class
        if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, class_names=self.class_names,
        grid_size=model_info_dict['grid_size'],
        point_cloud_range=model_info_dict['point_cloud_range'],
        voxel_size=model_info_dict['voxel_size'],
        predict_boxes_when_training=False)
    self.add_module("dense_head", dense_head)

  def forward(self, batch_size, voxels, voxel_num_points, voxel_coords):
    vfe_features = self.vfe(voxels, voxel_num_points)
    backbone_outputs = self.backbone_3d(batch_size=batch_size,
                                        vfe_features=vfe_features,
                                        voxel_coords=voxel_coords)

    head_outputs = self.dense_head()

    for cur_module in self.module_list:
      batch_dict = cur_module(batch_dict)

    if self.training:
      loss, tb_dict, disp_dict = self.get_training_loss()
      ret_dict = {
          'loss': loss
      }
      return ret_dict, tb_dict, disp_dict
    else:
      pred_dicts, recall_dicts = self.post_processing(batch_dict)
      return pred_dicts, recall_dicts

  def get_training_loss(self):

    disp_dict = {}
    loss, tb_dict = self.dense_head.get_loss()

    return loss, tb_dict, disp_dict

  def post_processing(self, batch_dict):
    post_process_cfg = self.model_cfg.POST_PROCESSING
    batch_size = batch_dict['batch_size']
    final_pred_dict = batch_dict['final_box_dicts']
    recall_dict = {}
    for index in range(batch_size):
      pred_boxes = final_pred_dict[index]['pred_boxes']

      recall_dict = self.generate_recall_record(
          box_preds=pred_boxes,
          recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
          thresh_list=post_process_cfg.RECALL_THRESH_LIST
      )

    return final_pred_dict, recall_dict
