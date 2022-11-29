_base_ = '../default.py'

expname = 'dvgo_hotdog'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./hw4_data/hotdog',
    dataset_type='blender',
    white_bkgd=True,
)
