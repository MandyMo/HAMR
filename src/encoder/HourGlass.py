
'''
    file:   HourGlass.py

    date:   2018_08_21
    author: zhangxiong(1025679612@qq.com)
    mark:   hourglass for hand pose
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter
import torch.optim as optim
import numpy as np
import math
import torchvision

Pool = nn.MaxPool2d

import sys
sys.path.append('./src')
from util import copy_state_dict

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, (1./n)**0.5)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.bn is not None:
            x = self.bn(x)
        return x

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class HGBlock(nn.Module):
    def __init__(self, n, f, bn=None, increase=128):
        super(HGBlock, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, f, 3, bn=bn)
        # Lower branch
        self.pool1 = Pool(2, 2)
        self.low1 = Conv(f, nf, 3, bn=bn)
        # Recursive hourglass
        if n > 1:
            self.low2 = HGBlock(n-1, nf, bn=bn)
        else:
            self.low2 = Conv(nf, nf, 3, bn=bn)
        self.low3 = Conv(nf, f, 3)      
        self.up2  = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

class HGEncoder(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, **kwargs):
        super(HGEncoder, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )

        self.features = nn.ModuleList([
            nn.Sequential(
                HGBlock(4, inp_dim, bn, increase),
                Conv(inp_dim, inp_dim, 3, bn=False),
                Conv(inp_dim, inp_dim, 3, bn=False)
            ) for i in range(nstack)
        ])

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack

    def forward(self, imgs):
        x = imgs
        x = self.pre(x)
        img_feature = x
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append( self.outs[i](feature) )

            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)

        return img_feature, torch.stack(preds, 1)

class FEncoder(nn.Module):
    def __init__(self, out_dim, inp_dim):
        super(FEncoder, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Sequential(
            Conv(inp_dim, inp_dim, 3, stride = 2, bn=True), # to 32 x 32
            Conv(inp_dim, inp_dim, 3, stride = 2, bn=True), # to 16 x 16
            Conv(inp_dim, inp_dim, 3, stride = 2, bn=True), # to 8 x 8
            Conv(inp_dim, inp_dim * 2, 3, stride = 2, bn=True), # to 4 x 4
            Conv(inp_dim * 2, inp_dim * 4, 3, stride = 2, bn=True), # to 2 x 2
            Conv(inp_dim * 4, out_dim, 3, stride = 2, bn=False, relu = False), # to 1 x 1
        )

    def forward(self, x):
        shape = x.shape
        batch_size = shape[0]
        assert shape[1] == self.inp_dim        
        return self.conv(x).reshape(batch_size, -1)

class Net(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn, increase=128, feature_count = 2048, **kwargs):
        super(Net, self).__init__()
        self.hg = HGEncoder(nstack = nstack, inp_dim = inp_dim, oup_dim = oup_dim, bn = bn, increase = increase)
        self.fe = FEncoder(inp_dim = inp_dim + oup_dim,  out_dim = feature_count)

    def forward(self, x):
        feature, heat_map = self.hg(x)
        feature = torch.cat((feature, heat_map[:, -1]), dim = 1)
        return heat_map, self.fe(feature)

def load_hourglass(nstack = 2, use_bn = False, pretrained = True, pretraind_path = ''):
    model = Net(nstack = nstack, inp_dim = 256, oup_dim = 64, bn = use_bn)
    if pretrained:
        copy_state_dict(model.state_dict(), torch.load(pretraind_path), prefix = 'module.', exclude_prefix = 'hg.')
    return model

if __name__ == '__main__':
    import cv2
    model = load_hourglass(nstack = 4, pretraind_path = 'pose_model.pkl', use_bn = False).cuda()
    img = torch.ones(
        [16, 3, 256, 256]
    ).float().cuda() 
    heat_map, feature = model(img)
    print(heat_map.shape)
    print(feature.shape)
    print('finished')
