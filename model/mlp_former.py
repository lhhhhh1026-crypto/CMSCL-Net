#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from timm.models.layers import DropPath
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)
# np.random.seed(0)


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous() # 32,1024,18,2*D
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    

class CAM(nn.Module):
    def __init__(self, channel, reduction=16,avg=True):
        super(CAM, self).__init__()
        if avg:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        output = torch.sum(x * y.expand_as(x),dim=1)
        return output
    
class crossAttentionLayer(nn.Module):
    def __init__(self, xchannel = 64, tchannel = 256):
        super(crossAttentionLayer, self).__init__()
        self.feature_size = tchannel
        # Linear transformations for Q, K, V from the same source
        self.query = nn.Linear(tchannel, tchannel,bias=False)
        self.key = nn.Linear(xchannel, tchannel,bias=False)
        self.value = nn.Linear(xchannel, tchannel,bias=False)

        self.scale = tchannel ** -0.5
        self.softmax = nn.Softmax(dim=1)
        # self.cadp = nn.Dropout(0.2)
        self.proj = nn.Sequential(
            nn.Linear(tchannel, tchannel, bias=False),
            nn.LayerNorm(tchannel),
            nn.ReLU(),
            # nn.Dropout(0.2)
            )
    def forward(self, x, y):
        # Apply linear transformations
        queries = self.query(x)
        keys = self.key(y).transpose(1, 2)
        values = self.value(y)
        # Scaled dot-product attention
        scores = queries @ keys * self.scale

        # Apply softmax
        attention_weights = self.softmax(scores)
        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return self.proj(output),attention_weights
    
class MultiHeadAttention(nn.Module):
    def __init__(self, in_ch, hid_ch, num_heads, att_drop=0.5, dropout=0.5, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.qkv_size = hid_ch // num_heads
        self.scale = self.qkv_size ** -0.5
        self.softmax = nn.Softmax(dim=-1)

        self.linear_qkv = nn.Linear(in_ch, 3*hid_ch, bias=bias)
        self.att_dropout = nn.Dropout(att_drop)
        self.proj = nn.Sequential(
            nn.Linear(hid_ch, in_ch, bias=False),
            nn.LayerNorm(in_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
            )

    def forward(self, x, attn_bias=None, mask=None):
        b, num, dim = x.size()
        qkv = self.linear_qkv(x).view(b, num, 3, self.num_heads, self.qkv_size).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1].transpose(3, 2), qkv[2]  # [b, h ,num, qkv_size]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        attn = q.matmul(k) * self.scale  # [b, h, q_len, k_len]


        a = self.softmax(attn)
        a = self.att_dropout(a)
        x = a.matmul(v)  # [b, h, q_len, c]

        x = x.transpose(1, 2).contiguous().view(
            b, num, -1)  # [b, q_len, h, attn]
        return self.proj(x)
    
class Former_branch(nn.Module):
    def __init__(self,feature_size=512,xchannel = 64):
        super(Former_branch,self).__init__()
        self.ca1 = crossAttentionLayer(xchannel=xchannel,tchannel=feature_size)
        # self.norm1 = nn.LayerNorm(feature_size)
        self.att = MultiHeadAttention(in_ch=feature_size,hid_ch=feature_size,num_heads=16,att_drop=0.,dropout=0.)
        # self.norm2 = nn.LayerNorm(feature_size)
        self.norm3 = nn.LayerNorm(feature_size)
        self.fc1 = nn.Sequential(
            nn.Linear(feature_size, feature_size*4, bias=False), 
            nn.ReLU(),
            nn.Linear(feature_size*4, feature_size, bias=False),
            )
        self.act = nn.ReLU()
        # self.fc2 = nn.Sequential(
        #     nn.Linear(feature_size, feature_size, bias=False), 
        #     nn.LayerNorm(feature_size),
        #     nn.ReLU(),
        #     # nn.Dropout(0.5),
        # )
        
        # self.drop_path = DropPath(0.) 

    def forward(self,t,x):
        '''
        x: b,n,c
        t: n,m,zdim
        '''
        #
        x = x.permute(0,2,1)
        z,_ = self.ca1(t,x)
        # z = self.norm1(z)
        z = self.att(z)+z
        z = self.act(self.norm3(self.fc1(z)))+z

        # z = z + self.drop_path(self.att(self.norm1(z)))
        # z = z + self.drop_path((self.fc1(self.norm2(z))))

        # z = self.att(z)+z
        # z = self.fc1(z)
        # z = self.fc2(self.fc1(z)+z)
        return z
class MLP_Former(nn.Module):
    def __init__(self, emb_dims=512,output_channels=40):
        super(MLP_Former, self).__init__()
        # self.args = args
        self.k = 10
        self.emb_dims = emb_dims
        self.token = nn.Parameter(torch.randn(1, 16, 512)).to('cuda')
        # torch.nn.init.trunc_normal_(self.token, std=0.02)
        torch.nn.init.orthogonal_(self.token)
        
        self.former_bn1 = nn.BatchNorm2d(64)
        self.former_bn2 = nn.BatchNorm2d(64)
        self.former_bn3 = nn.BatchNorm2d(128)
        self.former_bn4 = nn.BatchNorm2d(256)
        # self.former_bn5 = nn.BatchNorm1d(512)
        self.former_bn6 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.former_bn1,
                                   nn.ReLU())
        self.former1 = Former_branch(feature_size=emb_dims,xchannel=64)

        self.conv2 = nn.Sequential(nn.Conv2d((64)*2 , 64, kernel_size=1, bias=False),
                                   self.former_bn2,
                                   nn.ReLU())
        self.former2 = Former_branch(feature_size=emb_dims,xchannel=64)

        self.conv3 = nn.Sequential(nn.Conv2d((64)*2 , 128, kernel_size=1, bias=False),
                                   self.former_bn3,
                                   nn.ReLU())
        self.former3 = Former_branch(feature_size=emb_dims,xchannel=128)

        self.conv4 = nn.Sequential(nn.Conv2d((128)*2 , 256, kernel_size=1, bias=False),
                                   self.former_bn4,
                                   nn.ReLU())
        self.former4 = Former_branch(feature_size=emb_dims,xchannel=256)

        # self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
        #                            self.former_bn5,
        #                            nn.ReLU())
        
        self.former_linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.act = nn.ReLU()
        # self.dp = nn.Dropout(0.5)
        # self.claahead = nn.Linear(512,5)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)
        t = self.token.repeat(batch_size,1,1)
        # x ： 32*3*2048
        x = get_graph_feature(x, k=self.k)
        # x ： 32*6*2048*10
        x = self.conv1(x)
        # x ： 32*64*2048*10
        x1 = x.max(dim=-1, keepdim=False)[0]
        # x ： 32*64*2048
        z = self.former1(t,x1)

        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        z = self.former2(z,x2)

        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        z = self.former3(z,x3)

        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        z = self.former4(z,x4)
        # x = z.max(dim=1)[0]

        # x = torch.cat((x1, x2, x3, x4,), dim=1)
        # x = self.conv5(x)
        
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # print(x1.shape)
        # print(x2.shape)
        z1 = z.mean(dim = 1)
        z2 = z.max(dim=1)[0]
        # print(z1.shapea)
        # print(z2.shape)
        x = torch.cat((z1, z2), dim=-1)
        x = self.act(self.former_bn6(self.former_linear1(x)))
        return x