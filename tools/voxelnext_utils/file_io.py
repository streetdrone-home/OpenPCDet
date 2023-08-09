import numpy as np


def remove_ego_points(points, center_radius=1.0):
  mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
  return points[mask]


def get_sweep(root_path, sweep_info):
  lidar_path = root_path / sweep_info['lidar_path']
  points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
  points_sweep = remove_ego_points(points_sweep).T
  if sweep_info['transform_matrix'] is not None:
    num_points = points_sweep.shape[1]
    points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
        np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

  cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
  return points_sweep.T, cur_times.T


def get_lidar_with_sweeps(root_path, info, max_sweeps=1):
  lidar_path = root_path / info['lidar_path']
  points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

  sweep_points_list = [points]
  sweep_times_list = [np.zeros((points.shape[0], 1))]

  for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
    points_sweep, times_sweep = get_sweep(root_path, info['sweeps'][k])
    sweep_points_list.append(points_sweep)
    sweep_times_list.append(times_sweep)

  points = np.concatenate(sweep_points_list, axis=0)
  times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

  points = np.concatenate((points, times), axis=1)
  return points
