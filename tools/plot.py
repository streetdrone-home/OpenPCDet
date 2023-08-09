import torch
import numpy as np

from visual_utils import open3d_vis_utils as V

points = np.load('points.npy')
pred_dict = torch.load('pred_dicts1.pt')

print(points)

V.draw_scenes(
    points=points, ref_boxes=pred_dict['pred_boxes'],
    ref_scores=pred_dict['pred_scores'], ref_labels=pred_dict['pred_labels']
)
