import glob
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
    c2w[:, [1, 2]] *= -1
    return c2w


def load_nsvf_data(basedir):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], [], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)

    H, W = imgs[0].shape[:2]
    with open(os.path.join(basedir, 'intrinsics.txt')) as f:
        focal = float(f.readline().split()[0])

    R = np.sqrt((poses[..., :3, 3]**2).sum(-1)).mean()
    render_poses = torch.stack([pose_spherical(angle, -30.0, R)
                               for angle in np.linspace(-180, 180, 200 + 1)[:-1]], 0)

    return imgs, poses, render_poses, [H, W, focal], i_split
