import numpy as np


def angle(a, b):
    """# Get Angle Between Row Vectors"""
    from numpy import arctan2, pi

    theta_a = arctan2(a[:, 1], a[:, 0])
    theta_b = arctan2(b[:, 1], b[:, 0])
    d_theta = theta_b - theta_a
    assert (-pi <= d_theta) and (d_theta <= pi), "Theta difference outside of [-π,π]"
    return d_theta


def normalize(column):
    """# Normalize Column Vector"""
    from numpy.linalg import norm

    return column / norm(column, axis=0)


def winding(xy_grid, trajectory_start, trajectory_end):
    """# Get Winding Number on Grid"""
    from numpy import zeros, cross, clip, arcsin

    trajectories = trajectory_end - trajectory_start
    winding = zeros((xy_grid.shape[0]))
    for idx, trajectory in enumerate(trajectories):
        r = xy_grid - trajectory_start[idx]
        cross = cross(normalize(trajectory), normalize(r))
        cross = clip(cross, -1, 1)
        theta = arcsin(cross)
        winding += theta
    return winding


def vorticity(xy_grid, trajectory_start, trajectory_end):
    """# Get Vorticity Number on Grid"""
    from numpy import zeros, cross

    trajectories = trajectory_end - trajectory_start
    vorticity = zeros((xy_grid.shape[0]))
    for idx, trajectory in enumerate(trajectories):
        r = xy_grid - trajectory_start[idx]
        vorticity += cross(normalize(trajectory), normalize(r))
    return vorticity


def data_range(data):
    """# Get the range of values for each row"""
    from numpy import min, max

    return min(data, axis=0), max(data, axis=0)


np.normalize = normalize
np.range = data_range
