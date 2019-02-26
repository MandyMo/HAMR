
'''
    file:   util.py

    date:   2018_04_29
    author: zhangxiong(1025679612@qq.com)
'''

import h5py
import torch
import torch.nn as nn
import numpy as np
from config import args
import json
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import math
from scipy import interpolate

def batch_rodrigues(theta):
    batch_size = theta.shape[0]
    l1norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    
    return quat2mat(quat)

def quat2mat(quat):
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = Variable(torch.from_numpy(np_rot_x).float()).cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, Variable(torch.ones(N, 1, 1)).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)
    
    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, Variable(torch.zeros(N, 16, 1, 1)).cuda()], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_lrotmin(theta):
    theta = theta[:,3:].contiguous()
    Rs = batch_rodrigues(theta.reshape(-1, 3))
    e = Variable(torch.eye(3).float())
    Rs = Rs.sub(1.0, e)

    return Rs.reshape(-1, 23 * 9)

def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.reshape(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    return (camera[:, :, 0] * X_trans.reshape(shape[0], -1)).reshape(shape)

def batch_pers_proj(X, camera):
    '''
        X is N x num_points x 3
        Camera is N x 6(f, tx, ty, tz, dx, dy)
    '''
    camera = camera.reshape(-1, 1, args.cam_param_count)
    T, f, dxy = camera[:, :, 1:4], camera[:, :, 0:1], camera[:, :, 4:]
    X_trans = X + T
    Z = X_trans[:, :, 2:]
    X_projected = X_trans / Z
    return X_projected[:, :,:2] * f + dxy

def get_proj_func(proj_type = 'perspective'):
    return batch_pers_proj if proj_type == 'perspective' else batch_orth_proj

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = '', exclude_prefix = None):
    def _get_params(key):
        if exclude_prefix is not None:
            if key.startswith(exclude_prefix):
                key = key[len(exclude_prefix):]
        
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None
    
    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

def rod_vec_to_quat(rod):
    rod = rod.copy()
    radian = np.linalg.norm(rod)

    if radian <= 1e-6:
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    axis = rod / radian

    sv, cv = math.sin(radian / 2.0), math.cos(radian / 2.0)
    axis = axis * sv

    return np.array([axis[0], axis[1], axis[2], cv])
    

def mul_quat(qa, qb): #qa * qb
    va, vb = qa[:3], qb[:3]
    sa, sb = qa[3], qb[3]

    s = sa * sb - np.dot(va, vb)
    v = np.cross(vb, va) + sa * vb + sb * va
    return np.array([v[0], v[1], v[2], s])
    
def quat_to_rod_vec(q):
    radian = math.acos(q[3])
    sv = math.sin(radian)

    if math.fabs(sv) <= 1e-6:
        return np.array([0.0, 0.0, 0.0])

    axis = q[:3] / sv

    return axis * radian * 2

def mul_rod_vec(va, vb):
    qa, qb = rod_vec_to_quat(va), rod_vec_to_quat(vb)
    qr = mul_quat(qa, qb)
    return quat_to_rod_vec(qr)

def clamp(x, v_min, v_max):
    if x < v_min:
        return v_min
    if x > v_max:
        return v_max
    return x
