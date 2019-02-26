

'''
    file:   model.py

    date:   2018_07_31
    author: zhangxiong(1025679612@qq.com)
'''

import torch
import torch.nn as nn
import numpy as np
import mano
import config

from config import args
from LinearModel import LinearModel
from util import get_proj_func

from encoder.HourGlass import load_hourglass

import sys

class ThetaRegressor(LinearModel):
    def __init__(self, fc_layers, use_dropout, drop_prob, use_ac_func, iterations):
        super(ThetaRegressor, self).__init__(fc_layers, use_dropout, drop_prob, use_ac_func)
        self.iterations = iterations
        batch_size = args.batch_size

        init_cam_param = torch.ones((batch_size, args.cam_param_count))
        init_pose_param = torch.zeros((batch_size, 3 + args.pose_param_count))
        init_shape_param = torch.zeros((batch_size, args.shape_param_count))

        if args.predict_shape:
            init_param = torch.cat((init_cam_param, init_pose_param, init_shape_param), dim = 1)
        else:
            init_param = torch.cat((init_cam_param, init_pose_param), dim = 1)

        self.register_buffer('mean_theta', init_param.float())

    def forward(self, inputs):
        thetas = []
        shape = inputs.shape
        theta = self.mean_theta[:shape[0], :]
        for _ in range(self.iterations):
            total_inputs = torch.cat([inputs, theta], 1)
            theta = theta + self.fc_blocks(total_inputs)
            thetas.append(theta)
        return thetas

class ManoNetbase(nn.Module):
    def __init__(self):
        super(ManoNetbase, self).__init__()
        self.proj_func = get_proj_func(args.project_type)
        def _read_configs():
            self.encoder_name = args.encoder_name
            self.iterations = args.iterations
            self.feature_count = args.feature_count
            self.left_hand_model = args.left_hand_model
            self.right_hand_model = args.right_hand_model
            if args.predict_shape:
                self.total_theta_count = args.cam_param_count + 3 + args.pose_param_count + args.shape_param_count
            else:
                self.total_theta_count = args.cam_param_count + 3 + args.pose_param_count

        def _check_configs():
            self.encoder_name == 'hourglass':
            assert args.crop_size == 256
            assert args.feature_count == 2048

        def _create_network():
            def _create_hand_model():           
                self.hand_model = mano.ManoHand(
                    model_path = self.left_hand_model,
                    obj_saveable = True
                )

            def _create_regressor_model():
                fc_layers = [self.feature_count + self.total_theta_count, 1024, 1024, self.total_theta_count]
                use_dropout = [True, True, False]
                drop_prob = [0.5, 0.5, 0.5]
                use_ac_func = [True, True, False]

                self.regressor = ThetaRegressor(
                    fc_layers = fc_layers,
                    use_dropout = use_dropout,
                    drop_prob = drop_prob,
                    use_ac_func = use_ac_func,
                    iterations = self.iterations
                )

            def _create_encoder():
                self.encoder = load_hourglass(pretraind_path = 'pose_model.pkl')

            _create_hand_model()
            _create_regressor_model()
            _create_encoder()

        _read_configs()
        _check_configs()
        _create_network()

    def _calc_detail_info(self, theta):
        cam   = theta[:, 0 : args.cam_param_count]
        pose  = theta[:, args.cam_param_count : args.cam_param_count + 3 + args.pose_param_count]
        shape = theta[:, self.total_theta_count - args.shape_param_count:] if args.predict_shape else None
        verts, j3d, Rs = self.hand_model(beta = shape, theta = pose, get_skin = True)

        j2d = self.proj_func(j3d - j3d[:, 0:1, :], cam)
        return (theta, verts, j2d, j3d, Rs)

    def forward(self, input_images):
        ht, f = self.encoder(input_images)
        thetas = self.regressor(f)
        detail_info = self._calc_detail_info(thetas[-1])
        detail_info += (ht,)
        return [detail_info]
