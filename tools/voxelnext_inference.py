import numpy as np
import glob
import os
import open3d.ml.torch as ml3d  # or open3d.ml.tf as ml3d
from open3d.ml.vis import BoundingBox3D
import open3d

def process_data(root_path):
    data_dicts = []
    bounding_boxs_set = []
    paths = glob.glob(root_path)
    paths = sorted(glob.glob(root_path), key=lambda name: int(os.path.basename(name).split("_")[1]))

    # print(paths)
    # assert False
    for path in paths:
        data = np.load(path, allow_pickle=True).item()
        pre, filename = os.path.split(path)
        pre, _ = os.path.split(pre)
        pred_path = os.path.join(pre, "pred_frame8", filename)
        # print("pred_path", pred_path)
        pred_data = np.load(pred_path, allow_pickle=True).item()

        pred_box = pred_data["selected_boxes"][:, :7]
        # score = data["pred_scores"]
        # score_idx = np.argsort(-1*score)
        # top_k = min(10000, len(score_idx))
        # pred_box = pred_box[score_idx[:top_k]]
        labels = pred_data["selected_labels"]
        # pred_box = data["gt_boxes"][0, :, :7]
        name = os.path.basename(path).split(".")[0]
        points = data["voxels"][:, :3]
        bounding_boxs = []
        for i in range(pred_box.shape[0]):
            center = pred_box[i, :3]
            size = pred_box[i, 3:6]
            axis_angles = np.array([0, 0, pred_box[i, 6] + 1e-10])
            rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
            # front = np.array([0,0,1])
            # up = np.array([0,1,0])
            # left = np.array([1,0,0])
            front = rot[:, 2]
            up = rot[:, 1]
            left = rot[:, 0]
            # label_class = pred_box[i, -1]
            label_class = str(labels[i])
            if labels[i] != 10 and labels[i] !=9:
                confidence = 1
            else:
                # print("yes!!!!!!!!!!!!")
                confidence = -1
            bbox = BoundingBox3D(center,
                                 front,
                                 up,
                                 left,
                                 size,
                                 label_class,
                                 confidence=confidence,
                                 arrow_length=0)
            bounding_boxs.append(bbox)
        
        # center = pred_box[:, :3]
        # size = pred_box[:, 3:6]
        # front = np.ones((center.shape[0], 3))
        # up = np.ones((center.shape[0], 3))
        # left = np.ones((center.shape[0], 3))
        # confidence = np.ones(())
        # bbox = BoundingBox3D()


        data_dict = {}
        data_dict["points"] = points
        data_dict["name"] = name
        data_dict["bounding_boxes"] = bounding_boxs
        data_dict["labels"] = labels
        data_dicts.append(data_dict)
        # bounding_boxs_set.append(bounding_boxs)
    return data_dicts, bounding_boxs_set


if __name__ == "__main__":
    root_path = "Research/test_video_data/voxel_frame8/*.npy"
    data_dicts, bounding_boxs = process_data(root_path)
    # print(data_dicts)

    lut = ml3d.vis.LabelLUT()
    lut.add_label('1', 1, [1, 0, 0])
    lut.add_label('2', 2, [0, 0, 1])
    lut.add_label('3', 3, [0, 0, 1])
    lut.add_label('4', 4, [0, 0, 1])
    lut.add_label('5', 5, [0, 0, 1])
    lut.add_label('6', 6, [0, 0, 1])
    lut.add_label('7', 7, [0, 0, 1])
    lut.add_label('8', 8, [0, 0, 1])
    lut.add_label('9', 9, [0, 0, 1])

    vis = ml3d.vis.Visualizer()
    vis.visualize(data_dicts, bounding_boxes=None, lut=lut)
