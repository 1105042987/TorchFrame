from __future__ import absolute_import
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import numpy as np
import cv2
from docker.abstract_model import weak_evaluate, weak_loss

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 1, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=6,kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 1, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(1, 1, 3, 1, 0),
        )

    def forward(self, x):
        out = self.layer1(x[0])
        out = self.layer2(out)
        return out.view(-1,1)

class loss(weak_loss):
    def __init__(self):
        super(loss, self).__init__()
        self.L = nn.BCEWithLogitsLoss()

    def get_loss(self, pre, tar):
        if len(tar.shape)==4: return 0,{}
        loss = self.L(pre,tar.view(-1,1))
        return loss, {'BCE':loss}

class evaluate(weak_evaluate):
    def __init__(self, result_dir,image_shape,input_width):
        super(evaluate, self).__init__(result_dir)
        self.image_shape = [tuple(x) for x in image_shape]
        self.label_shape = [tuple(y//8 for y in x) for x in image_shape]
        self.input_width = input_width
        self.whole_image_len = 0
        self.label_size = [0]
        self.cnt=0
        self.kernel3 = np.ones((3, 3))
        self.kernel5 = np.ones((5, 5))
        self.kernelx = np.array([
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ])
        for box in self.label_shape:
            self.whole_image_len += box[0]*box[1]
            self.label_size.append(self.whole_image_len)

    def get_eval(self, inputs, preds, targets):
        if len(preds)!=self.whole_image_len:
            calc_position = (targets!=0.5)
            binary = (preds[calc_position]>0).long()
            tars = targets[calc_position]
            acc = (tars==binary).sum()/len(calc_pos)
            return {'acc':acc}
        preds[preds < 0] = 0
        label_list={}
        labelimg={}
        val=np.zeros(len(self.image_shape))
        for idx, box in enumerate(self.label_shape):
            pred = preds[self.label_size[idx]:self.label_size[idx+1]].detach().cpu().numpy()
            pred = pred.reshape(box[1], box[0])
            pred = cv2.filter2D(pred, -1, self.kernel3)
            pred = cv2.filter2D(pred, -1, self.kernelx)
            pred[pred < np.median(pred)] = 0
            pred[:2, :] = 0
            pred[-2:, :] = 0
            pred[:, :2] = 0
            pred[:, -2:] = 0
            find = cv2.dilate(pred, self.kernel5)
            find[(find != pred) | (pred == 0)] = 0
            val[idx] = np.sum(find > 0)
            if idx == 2 and val[idx] < 80:
                val[idx] = 0
            if idx == 1 and val[idx] < 15:
                val[idx] = 0
            find[find > 0] = 1
            label_list[idx] = np.argwhere(find == 1)
            find = cv2.filter2D(find, -1, np.ones((5, 5)))
            labelimg[idx] = find
            val[idx] -= np.sum(find[find > 1])/10
        idx = np.argmax(val)
        self.show_label = labelimg[idx]
        self.show_shape = self.image_shape[idx]
        self.label_list = label_list[idx]
        return {'num':val[idx]}

    def visualize(self, inputs, preds, targets, _eval):
        show = cv2.resize(targets[0].detach().cpu().numpy(), self.show_shape, interpolation=cv2.INTER_NEAREST)
        self._refreshGrid(show)
        cv2.imshow('labeled', show)
        cv2.moveWindow('labeled', 0, -30)
        key = cv2.waitKey(0)
        if key == ord('q'):
            raise('stop')
        return key

    def _refreshGrid(self, img):
        for row in range(0, img.shape[0]//8):
            for col in range(0, img.shape[1]//8):
                if self.show_label[row][col] == 0:
                    cv2.rectangle(img, (col*8,row*8), (col*8+8, row*8+8), (0,0,0), 1)
        for row in range(0, img.shape[0]//8):
            for col in range(0, img.shape[1]//8):
                if self.show_label[row][col] == 1:
                    cv2.rectangle(img, (col*8,row*8), (col*8+8, row*8+8), (0,255,0), 1)
                elif self.show_label[row][col] > 1:
                    cv2.rectangle(img, (col*8,row*8), (col*8+8, row*8+8), (0,255,255), 1)

    def save(self, inputs, preds, targets, _eval):
        ratio = targets.shape[3]/self.show_shape[1]*8
        pl = ((self.label_list+0.5)*ratio+0.5).astype(int)
        R = int(2.5*ratio+0.5)
        np.savez(self.result_dir+'{}.npz'.format(self.cnt), radius=R, data=pl, img=inputs[1])
        self.cnt+=1
        pass
