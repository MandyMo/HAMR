

'''
    file:   mano.py

    date:   2018_07_26
    author: zhangxiong(1025679612@qq.com)
'''

import torch
import torch.nn as nn
import numpy as np

import json
from util import batch_global_rigid_transformation, batch_rodrigues, batch_lrotmin
from config import args

class ManoHand(nn.Module):
    def __init__(self, model_path, obj_saveable = False):
        super(ManoHand, self).__init__()
        self.pose_param_count = args.pose_param_count

        self.finger_index = [734, 333, 443, 555, 678]

        batch_size = args.batch_size
        self.model_path = model_path
        with open(self.model_path) as reader:
            model = json.load(reader)
        if obj_saveable:
            self.faces = model['f']
        else:
            self.faces = None
    
        np_v_template = np.array(model['v_template'], dtype = np.float)
        vertex_count, vertex_component = np_v_template.shape[0], np_v_template.shape[1]
        self.size = [vertex_count, 3]

        if args.predict_shape:
            self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        else:
            np_v_template = np.tile(np_v_template, (batch_size, 1))
            self.register_buffer('v_template', torch.from_numpy(np_v_template).float().reshape(-1, vertex_count, vertex_component))
        
        np_J_regressor = np.array(model['J_regressor'], dtype = np.float).T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        num_shape_basis = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, num_shape_basis]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)

        np_weight = np.array(model['weights'], dtype = np.float)
        vertex_count, vertex_component = np_weight.shape[0], np_weight.shape[1]
        np_weight = np.tile(np_weight, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weight).float().reshape(-1, vertex_count, vertex_component))

        hands_mean = np.array(model['hands_mean'], dtype = np.float)
        self.register_buffer('hands_mean', torch.from_numpy(hands_mean).float())

        hands_components = np.array(model['hands_components'], dtype = np.float)
        self.register_buffer('hands_components', torch.from_numpy(hands_components).float())

        self.register_buffer('o1', torch.ones(batch_size, vertex_count).float())
        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.faces:
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, theta, get_skin = False):
        def _get_full_pose(theta):
            g_rot, partial_local_rot = theta[:, :3], theta[:, 3:]
            full_local_rot = torch.matmul(partial_local_rot, self.hands_components[:self.pose_param_count, :]) + self.hands_mean
            return torch.cat((g_rot, full_local_rot), dim = 1)

        if not self.cur_device:
            if theta is not None:
                device = theta.device
            else:
                device = beta.device
            self.cur_device = torch.device(device.type, device.index)
        
        theta = _get_full_pose(theta)
        num_batch = beta.shape[0] if beta is not None else theta.shape[0]
        if args.predict_shape:
            v_shaped = torch.matmul(beta, self.shapedirs).reshape(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template[:num_batch]

        Jx = torch.matmul(v_shaped[:,:,0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:,:,1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:,:,2], self.J_regressor)

        J = torch.stack([Jx, Jy, Jz], dim = 2)
        Rs = batch_rodrigues(theta.reshape(-1, 3)).reshape(-1, 16, 3, 3)
        pose_feature = Rs[:,1:,:,:].sub(1.0, self.e3).reshape(-1, 135)
        v_posed = torch.matmul(pose_feature, self.posedirs).reshape(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = False)

        weight = self.weight[:num_batch]
        W = weight.reshape(num_batch, -1, 16)
        T = torch.matmul(W, A.reshape(num_batch, 16, 16)).reshape(num_batch, -1, 4, 4)
        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:,:,:3,0]
        joint_x = torch.matmul(verts[:,:,0], self.J_regressor)
        joint_y = torch.matmul(verts[:,:,1], self.J_regressor)
        joint_z = torch.matmul(verts[:,:,2], self.J_regressor)

        finger_verts = verts[:, self.finger_index, :]

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        joints = torch.cat((joints, finger_verts), dim = 1)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints