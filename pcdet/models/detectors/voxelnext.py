from .detector3d_template import Detector3DTemplate
from pprint import pprint

class VoxelNeXt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # print(self.module_list)

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            # print("module #################################################################")
            batch_dict = cur_module(batch_dict)
            # pprint(cur_module)
            # print("\nbatch_dict: \n")
            # pprint(batch_dict)
            # print("#################################################################\n\n")

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

# class VoxelNeXt(Detector3DTemplate):
#     def __init__(self, model_cfg, num_class, dataset):
#         super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
#         self.module_list = self.build_networks()
#         # print(self.module_list)

#     def forward(self, voxels, voxel_coords, voxel_num_points, batch_size=1):

#         # for cur_module in self.module_list:
#         #     print("module #################################################################")
#         #     batch_dict = cur_module(batch_dict)
#         #     pprint(cur_module)
#         #     print("\nbatch_dict: \n")
#         #     pprint(batch_dict)
#         #     print("#################################################################\n\n")
        
#         voxel_features = self.module_list[0](voxels, voxel_num_points) # mean vfe
#         encoded_spconv_tensor = self.module_list[1](voxel_features, voxel_coords, batch_size=batch_size) # VoxelResBackBone8xVoxelNeXt
#         final_box_dicts = self.module_list[2](encoded_spconv_tensor, batch_size=batch_size) # VoxelNeXtHead

#         if self.training:
#             loss, tb_dict, disp_dict = self.get_training_loss()
#             ret_dict = {
#                 'loss': loss
#             }
#             return ret_dict, tb_dict, disp_dict
#         else:
#             return final_box_dicts

#     def get_training_loss(self):
        
#         disp_dict = {}
#         loss, tb_dict = self.dense_head.get_loss()
        
#         return loss, tb_dict, disp_dict

#     def post_processing(self, batch_dict):
#         post_process_cfg = self.model_cfg.POST_PROCESSING
#         batch_size = batch_dict['batch_size']
#         final_pred_dict = batch_dict['final_box_dicts']
#         recall_dict = {}
#         for index in range(batch_size):
#             pred_boxes = final_pred_dict[index]['pred_boxes']

#             recall_dict = self.generate_recall_record(
#                 box_preds=pred_boxes,
#                 recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
#                 thresh_list=post_process_cfg.RECALL_THRESH_LIST
#             )

#         return final_pred_dict, recall_dict