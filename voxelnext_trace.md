# VoxelNext trace

Input (batch_dict)
  - batch_size
  - frame_id
  - points
  - use_lead_xyz
  - voxel_coords
  - voxel_num_points
  - voxels

VFE (MeanVFE)
  - Input
    - voxels
    - voxel_num_points
  - Output
    - voxel_features

Backbone3D (VoxelResBackBone8xVoxelNeXt)
  - Input
    - voxel_features
    - voxel_coords
    - batch_size
  - Output
    - encoded_spconv_tensor
    - encoded_spconv_tensor_stride
    - multi_scale_3d_features
      - x_conv1
      - x_conv2
      - x_conv3
      - x_conv4
    - multi_scale_3d_strides
      - x_conv1
      - x_conv2
      - x_conv3
      - x_conv4

DenseHead (VoxelNeXtHead)
  - Input
    - encoded_spconv_tensor
    - batch_size
  - Output
    - rois
    - roi_scores
    - roi_labels
    - has_class_labels
    - final_box_dicts
      - pred_boxes
      - pred_ious
      - pred_labels
      - pred_scores

PostProcessing
  - Input
    - batch_size
    - final_box_dicts
      - pred_boxes
  - Output
    - [pred_dicts]
    - [recall_dicts]