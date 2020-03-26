import torch

class NMS:
    def __init__(self,threshold):
        self.th = threshold
    def __call__(self,bbox):
        

