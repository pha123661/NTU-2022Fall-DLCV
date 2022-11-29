import os
import sys
import copy
import time
import random
import argparse
from tqdm import tqdm, trange

import mmcv
import imageio
import numpy as np

import torch
import torch.nn.functional as F

from lib import utils, dvgo, dcvgo, dmpigo
from lib.load_data import load_data


def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--dump_images", action='store_true')

    # I/O argument
    parser.add_argument("--input_json", type=str, required=True,
                        help="the location of input json")
    parser.add_argument("--output_dir", type=str,
                        help="the directory to dump images")

    return parser


@torch.no_grad()
def render_viewpoints(model, render_poses, HW, Ks, ndc, render_kwargs, filenames,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False):
    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)

    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW / render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    rgbs = []
    depths = []
    bgmaps = []
    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []

    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        render_result_chunks = [
            {k: v for k, v in model(
                ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k]
                         for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()

        rgbs.append(rgb)
        depths.append(depth)
        bgmaps.append(bgmap)
        if i == 0:
            print('Testing', rgb.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)
            if eval_ssim:
                ssims.append(utils.rgb_ssim(rgb, gt_imgs[i], max_val=1))
            if eval_lpips_alex:
                lpips_alex.append(utils.rgb_lpips(
                    rgb, gt_imgs[i], net_name='alex', device=c2w.device))
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(
                    rgb, gt_imgs[i], net_name='vgg', device=c2w.device))

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')
        if eval_ssim:
            print('Testing ssim', np.mean(ssims), '(avg)')
        if eval_lpips_vgg:
            print('Testing lpips (vgg)', np.mean(lpips_vgg), '(avg)')
        if eval_lpips_alex:
            print('Testing lpips (alex)', np.mean(lpips_alex), '(avg)')

    if render_video_flipy:
        for i in range(len(rgbs)):
            rgbs[i] = np.flip(rgbs[i], axis=0)
            depths[i] = np.flip(depths[i], axis=0)
            bgmaps[i] = np.flip(bgmaps[i], axis=0)

    if render_video_rot90 != 0:
        for i in range(len(rgbs)):
            rgbs[i] = np.rot90(rgbs[i], k=render_video_rot90, axes=(0, 1))
            depths[i] = np.rot90(depths[i], k=render_video_rot90, axes=(0, 1))
            bgmaps[i] = np.rot90(bgmaps[i], k=render_video_rot90, axes=(0, 1))

    if savedir is not None and dump_images:
        print(f"Flushing image to disk {len(rgbs)}")
        for i, filename in zip(trange(len(rgbs)), filenames):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(savedir, f'{filename}.png')
            imageio.imwrite(filename, rgb8)

    rgbs = np.array(rgbs)
    depths = np.array(depths)
    bgmaps = np.array(bgmaps)

    return rgbs, depths, bgmaps


def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    '''Load images / poses / camera settings / data split.
    '''
    data_dict = load_data(cfg.data)

    # # construct data tensor
    # if data_dict['irregular_shape']:
    #     data_dict['images'] = [torch.FloatTensor(
    #         im, device='cpu') for im in data_dict['images']]
    # else:
    #     data_dict['images'] = torch.FloatTensor(
    #         data_dict['images'], device='cpu')
    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def _compute_bbox_by_cam_frustrm_bounded(cfg, HW, Ks, poses, i_train, near, far):
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        if cfg.data.ndc:
            pts_nf = torch.stack(
                [rays_o + rays_d * near, rays_o + rays_d * far])
        else:
            pts_nf = torch.stack(
                [rays_o + viewdirs * near, rays_o + viewdirs * far])
        xyz_min = torch.minimum(xyz_min, pts_nf.amin((0, 1, 2)))
        xyz_max = torch.maximum(xyz_max, pts_nf.amax((0, 1, 2)))
    return xyz_min, xyz_max


def _compute_bbox_by_cam_frustrm_unbounded(cfg, HW, Ks, poses, i_train, near_clip):
    # Find a tightest cube that cover all camera centers
    xyz_min = torch.Tensor([np.inf, np.inf, np.inf])
    xyz_max = -xyz_min
    for (H, W), K, c2w in zip(HW[i_train], Ks[i_train], poses[i_train]):
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w,
            ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        pts = rays_o + rays_d * near_clip
        xyz_min = torch.minimum(xyz_min, pts.amin((0, 1)))
        xyz_max = torch.maximum(xyz_max, pts.amax((0, 1)))
    center = (xyz_min + xyz_max) * 0.5
    radius = (center - xyz_min).max() * cfg.data.unbounded_inner_r
    xyz_min = center - radius
    xyz_max = center + radius
    return xyz_min, xyz_max


def compute_bbox_by_cam_frustrm(args, cfg, HW, Ks, poses, i_train, near, far, **kwargs):
    print('compute_bbox_by_cam_frustrm: start')
    if cfg.data.unbounded_inward:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_unbounded(
            cfg, HW, Ks, poses, i_train, kwargs.get('near_clip', None))

    else:
        xyz_min, xyz_max = _compute_bbox_by_cam_frustrm_bounded(
            cfg, HW, Ks, poses, i_train, near, far)
    print('compute_bbox_by_cam_frustrm: xyz_min', xyz_min)
    print('compute_bbox_by_cam_frustrm: xyz_max', xyz_max)
    print('compute_bbox_by_cam_frustrm: finish')
    return xyz_min, xyz_max


@torch.no_grad()
def compute_bbox_by_coarse_geo(model_class, model_path, thres):
    print('compute_bbox_by_coarse_geo: start')
    eps_time = time.time()
    model = utils.load_model(model_class, model_path)
    interp = torch.stack(torch.meshgrid(
        torch.linspace(0, 1, model.world_size[0]),
        torch.linspace(0, 1, model.world_size[1]),
        torch.linspace(0, 1, model.world_size[2]),
    ), -1)
    dense_xyz = model.xyz_min * (1 - interp) + model.xyz_max * interp
    density = model.density(dense_xyz)
    alpha = model.activate_density(density)
    mask = (alpha > thres)
    active_xyz = dense_xyz[mask]
    xyz_min = active_xyz.amin(0)
    xyz_max = active_xyz.amax(0)
    print('compute_bbox_by_coarse_geo: xyz_min', xyz_min)
    print('compute_bbox_by_coarse_geo: xyz_max', xyz_max)
    eps_time = time.time() - eps_time
    print('compute_bbox_by_coarse_geo: finish (eps time:', eps_time, 'secs)')
    return xyz_min, xyz_max


def create_new_model(cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path):
    model_kwargs = copy.deepcopy(cfg_model)
    num_voxels = model_kwargs.pop('num_voxels')
    if len(cfg_train.pg_scale):
        num_voxels = int(num_voxels / (2**len(cfg_train.pg_scale)))

    if cfg.data.ndc:
        print(
            f'scene_rep_reconstruction ({stage}): \033[96muse multiplane images\033[0m')
        model = dmpigo.DirectMPIGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    elif cfg.data.unbounded_inward:
        print(
            f'scene_rep_reconstruction ({stage}): \033[96muse contraced voxel grid (covering unbounded)\033[0m')
        model = dcvgo.DirectContractedVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            **model_kwargs)
    else:
        print(
            f'scene_rep_reconstruction ({stage}): \033[96muse dense voxel grid\033[0m')
        model = dvgo.DirectVoxGO(
            xyz_min=xyz_min, xyz_max=xyz_max,
            num_voxels=num_voxels,
            mask_cache_path=coarse_ckpt_path,
            **model_kwargs)
    model = model.to(device)
    optimizer = utils.create_optimizer_or_freeze_model(
        model, cfg_train, global_step=0)
    return model, optimizer


def load_existed_model(args, cfg, cfg_train, reload_ckpt_path):
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, reload_ckpt_path).to(device)
    optimizer = utils.create_optimizer_or_freeze_model(
        model, cfg_train, global_step=0)
    model, optimizer, start = utils.load_checkpoint(
        model, optimizer, reload_ckpt_path, args.no_reload_optimizer)
    return model, optimizer, start


def scene_rep_reconstruction(args, cfg, cfg_model, cfg_train, xyz_min, xyz_max, data_dict, stage, coarse_ckpt_path=None):
    # init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if abs(cfg_model.world_bound_scale - 1) > 1e-9:
        xyz_shift = (xyz_max - xyz_min) * (cfg_model.world_bound_scale - 1) / 2
        xyz_min -= xyz_shift
        xyz_max += xyz_shift
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]

    # find whether there is existing checkpoint path
    last_ckpt_path = os.path.join(
        cfg.basedir, cfg.expname, f'{stage}_last.tar')
    if args.no_reload:
        reload_ckpt_path = None
    elif args.ft_path:
        reload_ckpt_path = args.ft_path
    elif os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    # init model and optimizer
    if reload_ckpt_path is None:
        print(f'scene_rep_reconstruction ({stage}): train from scratch')
        model, optimizer = create_new_model(
            cfg, cfg_model, cfg_train, xyz_min, xyz_max, stage, coarse_ckpt_path)
        start = 0
        if cfg_model.maskout_near_cam_vox:
            model.maskout_near_cam_vox(poses[i_train, :3, 3], near)
    else:
        print(
            f'scene_rep_reconstruction ({stage}): reload from {reload_ckpt_path}')
        model, optimizer, start = load_existed_model(
            args, cfg, cfg_train, reload_ckpt_path)

    # init rendering setup
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }


if __name__ == '__main__':

    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # cfg['data']['datadir'] = os.path.dirname(args.input_json)
    cfg['data']['input_json'] = args.input_json
    cfg['data']['dataset_type'] = 'blender'
    cfg['data']['white_bkgd'] = True

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    seed_everything()

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    # load model for rendring
    if args.render_test or args.render_train or args.render_video:
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
        ckpt_name = ckpt_path.split('/')[-1][:-4]
        if cfg.data.ndc:
            model_class = dmpigo.DirectMPIGO
        elif cfg.data.unbounded_inward:
            model_class = dcvgo.DirectContractedVoxGO
        else:
            model_class = dvgo.DirectVoxGO
        model = utils.load_model(model_class, ckpt_path).to(device)
        stepsize = cfg.fine_model_and_render.stepsize
        render_viewpoints_kwargs = {
            'model': model,
            'ndc': cfg.data.ndc,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_depth': True,
            },
        }

    # render testset and eval
    if args.render_test:
        os.makedirs(args.output_dir, exist_ok=True)
        print('All results wil be dumped into', args.output_dir)
        rgbs, depths, bgmaps = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            savedir=args.output_dir,
            dump_images=args.dump_images,
            filenames=data_dict['filenames'],
            **render_viewpoints_kwargs
        )

    print('Done')
