

'''
    file:   config.py

    date:   2018_07_26
    author: zhangxiong(1025679612@qq.com)
'''

import argparse
import os

parser = argparse.ArgumentParser(description = 'hand mesh recover model')

BASE_EXE_PATH = '/usr/mano/'

parser.add_argument(
    '--encoder-name',
    type = str,
    default = 'hourglass',
    help = 'encoder network name'
)

parser.add_argument(
    '--use-cam-loss',
    type = bool,
    default = True,
    help = 'use cam loss or not.'
)

parser.add_argument(
    '--train-model',
    type = str,
    default = 'hand', #'hand' or 'pose'
    help = 'train hand model or pose submodel.'
)

parser.add_argument(
    '--debug',
    type = bool,
    default = False,
    help = 'is debug mode or not.'
)

parser.add_argument(
    '--use-heatmap',
    type = bool,
    default = True,
    help = 'use heatmap or not.'
)

parser.add_argument(
    '--heatmap-resolution',
    type = int,
    default = 64,
    help = 'the heatmap resolution.'
)

parser.add_argument(
    '--heatmap-sigma',
    type = float,
    default = 2.5,
    help = 'the sigma of gaussion distribution'
)

parser.add_argument(
    '--save-folder',
    type = str,
    default = os.path.join(BASE_EXE_PATH, 'trained_model'),
    help = 'save trained model.'
)

parser.add_argument(
    '--reg-parameter',
    type = bool,
    default = False,
    help = 'use parameter regularization or not.'
)

parser.add_argument(
    '--batch-size',
    type = int,
    default = 128,
    help = 'batch size'
)

parser.add_argument(
    '--left-hand-model',
    type = str,
    default = os.path.join(BASE_EXE_PATH, 'model', 'left.txt'),
    help = 'left hand model path'
)

parser.add_argument(
    '--right-hand-model',
    type = str,
    default = os.path.join(BASE_EXE_PATH, 'model', 'right.txt'),
    help = 'right han model path'
)

parser.add_argument(
    '--cam-param-count',
    type = int,
    default = 4,
    help = 'cam component count'
)

parser.add_argument(
    '--shape-param-count',
    type = int,
    default = 10,
    help = 'shape component count'
)

parser.add_argument(
    '--pose-param-count',
    type = int,
    default = 12,
    help = 'pose component count'
)

parser.add_argument(
    '--predict-camera',
    type = bool,
    default = True,
    help = 'predict camera parameter or not.'
)

parser.add_argument(
    '--predict-shape',
    type = bool,
    default = True,
    help = 'predict the shape\'s hand or not.'
)

parser.add_argument(
    '--iterations',
    type = int,
    default = 3,
    help = 'default iterations'
)

parser.add_argument(
    '--num-worker',
    type = int,
    default = 12,
    help = 'the number worker'
)

parser.add_argument(
    '--e-lr',
    type = float,
    default = 0.0002,
    help = 'generator learning rate.'
)

parser.add_argument(
    '--pose-lr',
    type = float,
    default = 2e-4,
    help = 'pose submodel traing learning rate.'
)

parser.add_argument(
    '--gamma',
    type = float,
    default = 0.3,
    help = 'weight decay gamma.'
)

parser.add_argument(
    '--iter-count',
    type = int,
    default = 2000000,
    help = 'iter count of training.'
)

parser.add_argument(
    '--w-param-reg',
    type = float,
    default = 1e-6,
    help = 'weight of parameter regularization.'
)

parser.add_argument(
    '--w-loss-ht',
    type = float,
    default = 1000,
    help = 'weight of heatmap loss.'
)

parser.add_argument(
    '--w-loss-cam',
    type = float,
    default = 5e-4,
    help = 'weight of camera loss.'
)

parser.add_argument(
    '--w-loss-kp-2d',
    type = float,
    default = 1,
    help = 'weight of 2d keypoint loss.'
)

parser.add_argument(
    '--w-loss-kp-3d',
    type = float,
    default = 1000,
    help = 'weight of 3d keypoint loss.'
)

parser.add_argument(
    '--project-type',
    type = str,
    default = 'orthogonal', #['perspective', 'orthogonal']
    help = 'the projection type of 3d to 2d'
)

crop_size = {
    'hourglass' : 256
}

feature_count = {
    'hourglass' : 2048,
}

encoder_decay = {
    'hourglass' : 1e-5
}

cam_param_count = {
    'perspective' : 6,
    'orthogonal' : 3,
}

pretrained_model = {
    'generator' : os.path.join(BASE_EXE_PATH, 'fine_tuned', 'model.pkl')
}

args = parser.parse_args()
args.feature_count    = feature_count[args.encoder_name]
args.crop_size        = crop_size[args.encoder_name]
args.e_wd             = encoder_decay[args.encoder_name]
args.cam_param_count  = cam_param_count[args.project_type]