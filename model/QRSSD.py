from __future__ import absolute_import
import sys
sys.path.append('..')
import cv2
import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from docker.abstract_model import weak_evaluate, weak_loss
from utils.loss.focal_loss import FocalLoss2
from utils.utils import calcAUC
class net(nn.Module):
    def __init__(self, input_shape=(504,672),numMultiScale=4, numClass=1):
        super(net, self).__init__()
        self.flod = torch.nn.Unfold((8, 8), padding=0, stride=8)
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1,kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(1, 6, 3, 2, 0),
            nn.ReLU(),
            nn.Conv2d(6, 1, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 0),
            nn.ReLU(),
        )
        self.downSample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 1, 3, 2, 1),
                nn.ReLU()
            ) for i in range(numMultiScale-1)
        ])
        self.loc = nn.ModuleList([
            nn.Sequential( # delta_h, delta_w
                nn.Conv2d(1, 2, 3, 1, 1),
            ) for i in range(numMultiScale)
        ])
        self.numClass = numClass
        self.conf = nn.ModuleList([
            nn.Sequential( # confidence
                nn.Conv2d(1, numClass, 3, 1, 1),
            ) for i in range(numMultiScale)
        ])
        # self.const_bias = []
        # s = (int(input_shape[0])//8,int(input_shape[1])//8)
        # for i in range(numMultiScale):
        #     print(s)
        #     self.const_bias.append(
        #         torch.cat([
        #             torch.arange(s[0]).unsqueeze(1).repeat(1,1,s[1])+0.5,
        #             torch.arange(s[1]).unsqueeze(0).repeat(1,s[0],1)+0.5,
        #             torch.zeros(1,s[0],s[1])
        #         ]).float()
        #     )
        #     s = ((s[0]+1)//2,(s[1]+1)//2)

    def forward(self, x):
        B,C,H,W = x.shape
        x = self.flod(x).permute(0, 2, 1).contiguous().view(-1, 3, 8, 8)
        x = self.feature(x).reshape(B,1,H//8,W//8)          # 95ms abrove
        pos = [self.loc[0](x).reshape(B,2,-1)]
        pro = [self.conf[0](x).reshape(B,self.numClass,-1)]
        for down,loc,conf in zip(self.downSample,self.loc[1:],self.conf[1:]):
            x = down(x)
            pos.append(loc(x).reshape(B,2,-1))
            pro.append(conf(x).reshape(B,self.numClass,-1))
        return torch.cat(pos,-1),torch.cat(pro,-1)


class loss(weak_loss):
    def __init__(self,numMultiScale=4):
        super(loss, self).__init__()
        self.eye=torch.Tensor([
            [1.0, 0.3, 0.0, 0.0],
            [0.3, 1.0, 0.3, 0.0],
            [0.0, 0.3, 1.0, 0.3],
            [0.0, 0.0, 0.3, 1.0],
        ])
        # self.eye = torch.eye(4)
        self.shapes = []
        self.starts = []
        s = (504//8,672//8)
        self.totalNum=0
        for i in range(numMultiScale):
            self.starts.append(self.totalNum)
            self.shapes.append(s)
            self.totalNum+=s[0]*s[1]
            s = ((s[0]+1)//2,(s[1]+1)//2)
        self.class_loss = FocalLoss2(alpha=0.02,gamma=5,logits=True)
        self.bbox_loss = nn.SmoothL1Loss()

    def get_loss(self, pre, tar):
        bbox,prob = pre[0].permute(0,2,1), pre[1][:,0,:]
        gt = self.encodeGT(tar)
        pos_ind = gt[...,2] != 0
        bboxGT = gt[pos_ind][...,:2]
        bboxPre = F.sigmoid(bbox[pos_ind])
        bboxLoss = self.bbox_loss(bboxPre,bboxGT)
        # classLoss = self.class_loss(prob.reshape(-1,2),gt[...,2].long())
        classLoss = self.class_loss(prob,gt[...,2])
        return classLoss, {
            'sum':bboxLoss+classLoss,
            'bbox':bboxLoss,
            'class':classLoss,
            'auc':calcAUC(prob,pos_ind.float()),
        }
    
    def encodeGT(self,tar):
        # tar(B,100,4)
        numBBox = tar[1]                                    # (B,)
        tar = tar[0]
        B = tar.shape[0]
        GT = torch.zeros(B,self.totalNum,3).to(tar.device)  # (B,N,3)
        c_hw = (tar[...,:2]+tar[...,2:])/2
        size = torch.log2((tar[...,2:]-tar[...,:2]).max(-1)[0]/24)
        size[size<0]=-1.5
        size = self.eye[size.long()+1].to(tar.device)       # (B,100,4)
        p = (c_hw/8).float()                                # (B,100,2)
        n=0
        for s,b in zip(self.shapes,self.starts):
            pos = p.long()[...,0]*s[1]+p.long()[...,1]+b    # (B,100)
            val = (p-p.long().float())                      # (B,100,2)
            for i in range(B):
                v = numBBox[i]  # valid
                GT[i,pos[i,:v],:2] = val[i,:v,:]
                GT[i,pos[i,:v],2] = size[i,:v,n]
            p=p/2
            n+=1
        return GT
    def checkGT(self,tar,gt):
        # gt(B,N,3)
        score = gt[0,...,2]
        index = tar[1][0]
        ground = tar[0][0,...]
        radius = 8
        for s,b in zip(self.shapes,self.starts):
            num = s[0]*s[1]
            temp = score[b:b+num].reshape(s[0],s[1])
            
        



class evaluate(weak_evaluate):
    def __init__(self,threshold=0.5):
        super(evaluate, self).__init__()
        self.th = threshold

    def get_eval(self, inputs, preds, targets):
        find = cv2.dilate(pre, self.kernel5)
        find[(find != pre) | (pre == 0)] = 0
        return {}

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass

