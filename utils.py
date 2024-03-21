import glob

import numpy as np


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def sample_pcl(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    return pcl[pt_indices]


def estimate_center_scale(pcl):
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    return [center, scale]


def standardize_pc(pcl, center, scale):
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


def standardize_bbox(pcl, points_per_object):
    pcl = sample_pcl(pcl, points_per_object)
    center, scale = estimate_center_scale(pcl)
    return standardize_pc(pcl, center, scale)


def get_files(pattern):
    return glob.glob(pattern)
