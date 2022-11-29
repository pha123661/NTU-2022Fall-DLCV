import json
import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F


def trans_t(t): return torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()


def rot_phi(phi): return torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()


def rot_theta(th): return torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(input_json, half_res=False, testskip=1):

    with open(input_json, 'r') as fp:
        meta = json.load(fp)
    filenames = [os.path.basename(x['file_path']) for x in meta['frames']]

    if testskip == 0:
        skip = 1
    else:
        skip = testskip
    poses = []
    for frame in meta['frames'][::skip]:
        poses.append(np.array(frame['transform_matrix']))
    # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)
    i_split = [0, 0, list(range(len(poses)))]
    H, W = 800, 800
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0)
                               for angle in np.linspace(-180, 180, 160 + 1)[:-1]], 0)

    return None, poses, render_poses, [H, W, focal], i_split, filenames
