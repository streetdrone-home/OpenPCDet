import torch
import torch.nn as nn
import os
from .ops.iou3d_nms import iou3d_nms_utils
from .modules.mean_vfe import MeanVFE
from .modules.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .modules.voxelnext_head import VoxelNeXtHead
from .utils.spconv_utils import find_all_spconv_keys


class VoxelNext(nn.Module):
  def __init__(
      self, model_cfg, class_names, num_point_features, grid_size, point_cloud_range,
          voxel_size) -> None:
    super().__init__()
    self.class_names = class_names
    self.num_class = len(self.class_names)
    self.register_buffer('global_step', torch.LongTensor(1).zero_())
    self.model_cfg = model_cfg
    self._build_network(num_point_features, grid_size, point_cloud_range, voxel_size)

  @property
  def mode(self):
    return 'TRAIN' if self.training else 'TEST'

  def update_global_step(self):
    self.global_step += 1

  def _build_network(self, num_point_features, grid_size, point_cloud_range, voxel_size):
    # voxel feature encoder
    self.vfe = MeanVFE(num_point_features)
    num_point_features = self.vfe.get_output_feature_dim()

    # backbone 3d
    self.backbone_3d = VoxelResBackBone8xVoxelNeXt(
        self.model_cfg, input_channels=num_point_features,
        grid_size=grid_size)

    # dense head
    self.dense_head = VoxelNeXtHead(
        self.model_cfg.DENSE_HEAD, num_class=self.num_class
        if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1, class_names=self.class_names,
        grid_size=grid_size,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        predict_boxes_when_training=False)

  def forward(self, batch_size, voxels, voxel_num_points, voxel_coords):
    vfe_features = self.vfe(voxels, voxel_num_points)
    encoded_spconv_tensor = self.backbone_3d(batch_size=batch_size,
                                             vfe_features=vfe_features,
                                             voxel_coords=voxel_coords)

    head_outputs = self.dense_head(batch_size, encoded_spconv_tensor, gt_boxes=None)

    if self.training:
      loss, tb_dict, disp_dict = self.get_training_loss()
      ret_dict = {
          'loss': loss
      }
      return ret_dict, tb_dict, disp_dict
    else:
      pred_dicts, recall_dicts = self.post_processing(batch_size, head_outputs)
      return pred_dicts, recall_dicts

  def get_training_loss(self):

    disp_dict = {}
    loss, tb_dict = self.dense_head.get_loss()

    return loss, tb_dict, disp_dict

  @staticmethod
  def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
    if 'gt_boxes' not in data_dict:
      return recall_dict

    rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
    gt_boxes = data_dict['gt_boxes'][batch_index]

    if recall_dict.__len__() == 0:
      recall_dict = {'gt': 0}
      for cur_thresh in thresh_list:
        recall_dict['roi_%s' % (str(cur_thresh))] = 0
        recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

    cur_gt = gt_boxes
    k = cur_gt.__len__() - 1
    while k >= 0 and cur_gt[k].sum() == 0:
      k -= 1
    cur_gt = cur_gt[:k + 1]

    if cur_gt.shape[0] > 0:
      if box_preds.shape[0] > 0:
        iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
      else:
        iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

      if rois is not None:
        iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

      for cur_thresh in thresh_list:
        if iou3d_rcnn.shape[0] == 0:
          recall_dict['rcnn_%s' % str(cur_thresh)] += 0
        else:
          rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
          recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
        if rois is not None:
          roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
          recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

      recall_dict['gt'] += cur_gt.shape[0]
    else:
      gt_iou = box_preds.new_zeros(box_preds.shape[0])
    return recall_dict

  def post_processing(self, batch_size, out_dict):
    post_process_cfg = self.model_cfg.POST_PROCESSING
    final_pred_dict = out_dict['final_box_dicts']
    recall_dict = {}
    for index in range(batch_size):
      pred_boxes = final_pred_dict[index]['pred_boxes']
      recall_dict = self.generate_recall_record(
          box_preds=pred_boxes,
          recall_dict=recall_dict, batch_index=index, data_dict=out_dict,
          thresh_list=post_process_cfg.RECALL_THRESH_LIST
      )

    return final_pred_dict, recall_dict

  def _load_state_dict(self, model_state_disk, *, strict=True):
    state_dict = self.state_dict()  # local cache of state_dict

    spconv_keys = find_all_spconv_keys(self)

    update_model_state = {}
    for key, val in model_state_disk.items():
      if key in spconv_keys and key in state_dict and state_dict[key].shape != val.shape:
        # with different spconv versions, we need to adapt weight shapes for spconv blocks
        # adapt spconv weights from version 1.x to version 2.x if you used weights from spconv 1.x

        val_native = val.transpose(-1, -2)  # (k1, k2, k3, c_in, c_out) to (k1, k2, k3, c_out, c_in)
        if val_native.shape == state_dict[key].shape:
          val = val_native.contiguous()
        else:
          assert val.shape.__len__() == 5, 'currently only spconv 3D is supported'
          # (k1, k2, k3, c_in, c_out) to (c_out, k1, k2, k3, c_in)
          val_implicit = val.permute(4, 0, 1, 2, 3)
          if val_implicit.shape == state_dict[key].shape:
            val = val_implicit.contiguous()

      if key in state_dict and state_dict[key].shape == val.shape:
        update_model_state[key] = val
        # logger.info('Update weight %s: %s' % (key, str(val.shape)))

    if strict:
      self.load_state_dict(update_model_state)
    else:
      state_dict.update(update_model_state)
      self.load_state_dict(state_dict)
    return state_dict, update_model_state

  def load_params_from_file(self, filename, logger, to_cpu=False, pre_trained_path=None):
    if not os.path.isfile(filename):
      raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' %
                (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['model_state']
    if not pre_trained_path is None:
      pretrain_checkpoint = torch.load(pre_trained_path, map_location=loc_type)
      pretrain_model_state_disk = pretrain_checkpoint['model_state']
      model_state_disk.update(pretrain_model_state_disk)

    version = checkpoint.get("version", None)
    if version is not None:
      logger.info('==> Checkpoint trained from version: %s' % version)

    state_dict, update_model_state = self._load_state_dict(model_state_disk, strict=False)

    for key in state_dict:
      if key not in update_model_state:
        logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(state_dict)))
