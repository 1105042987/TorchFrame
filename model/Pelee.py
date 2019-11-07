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

    def forward(self, xs):
        pass

class loss(weak_loss):
    def __init__(self):
        super(loss, self).__init__()
    def get_loss(self, pre,tar):
        return None, {}

class evaluate(weak_evaluate):
    def __init__(self, result_dir):
        super(evaluate, self).__init__(result_dir)

    def get_eval(self, inputs, preds, targets):
        return {}

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass
