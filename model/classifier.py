from __future__ import absolute_import
import sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import os
import cv2
import torch 
import importlib
import numpy as np
import torch.nn as nn
from docker.abstract_model import weak_evaluate,weak_loss

class net(nn.Module):
    def __init__(self, backbone_position, param):
        super(net, self).__init__()
        module = importlib.import_module(backbone_position[0])
        self.NET = getattr(module,backbone_position[1])(**param)
        
    def forward(self, xs):
        return self.NET(xs)

###################################################### digest
# class loss(weak_loss):
#     def __init__(self):
#         super(loss, self).__init__()
#         self.L = nn.BCEWithLogitsLoss()
#     def get_loss(self, pre,tar):
#         loss = self.L(pre,tar)
#         return loss, {'CE':loss}

# class evaluate(weak_evaluate):
#     def __init__(self, th):
#         super(evaluate, self).__init__()
#         self.s = nn.Sigmoid()
#         self.th = th

#     def get_eval(self, inputs, preds, targets):
#         sig_p = self.s(preds)
#         acc = ((sig_p > self.th).long()==targets.long()).float().mean()
#         TP = ((sig_p > self.th).long()[targets == 1] == 1).sum()
#         FP = ((sig_p > self.th).long()[targets == 1] == 0).sum()
#         TN = ((sig_p > self.th).long()[targets == 0] == 0).sum()
#         FN = ((sig_p > self.th).long()[targets == 0] == 1).sum()
#         self.error_idx = (self.s(preds)>self.th).long()!=targets.long()
#         AllRight = float(sig_p.mean()>self.th)==targets[0]
#         return {'acc':acc,'TP':TP,'FP':FP,'TN':TN,'FN':FN,'AllRight':AllRight}

#     def visualize(self, inputs, preds, targets, _eval):
#         name = inputs[1][self.error_idx]
#         if len(name)==0:
#             return 'c'
#         true = targets[self.error_idx]
#         for n,l in zip(name,true):
#             img = cv2.imread(n)
#             cv2.imshow(str(l==1),img)
#             key = cv2.waitKey(0)
#             if key == ord('q'):
#                 raise('Quit')
#         return key

#     def save(self, inputs, preds, targets, _eval):
#         torch.save({
#             'patches':self.s(preds),
#             'targets':targets,
#             'img':targets.sum()>0,
#         },self.result_dir+os.path.basename(inputs[1]).replace('.jpg','.pth'))


################################################################ PANDA
class loss(weak_loss):
    def __init__(self):
        super(loss, self).__init__()
        self.L = nn.CrossEntropyLoss()
    def get_loss(self, pre,tar):
        loss = self.L(pre,tar)
        return loss, {'CE':loss}


class evaluate(weak_evaluate):
    def __init__(self, whole_img):
        from sklearn.metrics import cohen_kappa_score
        super(evaluate, self).__init__()
        self.whole_img = whole_img
        self.qwk = cohen_kappa_score
        self.his_tar = []
        self.his_pre = []

    def get_eval(self, inputs, preds, targets):
        if self.whole_img:
            self.his_pre.append(preds.mean(0).argmax().detach().cpu())
            self.his_tar.append(targets[0].detach().cpu())
            return {'PictureRight':preds.mean(0).argmax()==targets[0]}
        preds=preds.argmax(-1).float()
        judge = preds==targets.float()
        dic = {}
        dic['acc'] = judge.float().mean()
        for i in range(6):
            dic['L{}_T'.format(i)] = judge[targets.long()==i].sum()
            dic['L{}'.format(i)] = (targets.long()==i).sum()
        return dic

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass
    
    def final(self):
        if self.whole_img:
            return self.qwk(np.array(self.his_pre),np.array(self.his_tar),weights='quadratic')


if __name__ == "__main__":
    net_param={
        "backbone_position":["utils.backbone.Efficientnet","Efficientnet"],
        "param":{
            "pretrained":True,
            "name":"efficientnet-b0",
            "num_classes":6
        }
    }
    n = net(**net_param)
    print(n([torch.ones(5,3,256,256)]))