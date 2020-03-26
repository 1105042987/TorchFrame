import torchvision.transforms as T
from torch.utils.data.sampler import WeightedRandomSampler
from docker.abstract_model import weak_SplitPatch
import torch
import cv2
import os

class Digestpath_fixed_list(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transforms):
        super(Digestpath_fixed_list, self).__init__()
        import pandas as pd
        import numpy as np
        self.base = cfg['direction']
        self.mode = mode
        self.train_rate = cfg['train_rate']
        self.whole = cfg['whole_image']
        self.batch = cfg['batch_size']
        self.pure_pos = cfg['pure_pos']
        self.patch_size = cfg['patch_size']
        self.stride = int(self.patch_size*cfg['stride_rate'])
        self.positive_picture_only = cfg['positive_picture_only']
        rand_list = torch.load(self.base+cfg['shuffle_name'])
        rand_list_neg = np.array(rand_list['neg'])
        rand_list_pos = np.array(rand_list['pos'])
        name = torch.load(self.base+'sorted_namelist.pt')
        name['neg'] = np.array(name['neg'])
        name['pos'] = np.array(name['pos'])
        if mode == 'train':
            rand_list_neg = rand_list_neg[:int(len(rand_list_neg)*self.train_rate)]
            rand_list_pos = rand_list_pos[:int(len(rand_list_pos)*self.train_rate)]
            self.neg = [self.base+'patch/{}/{}'.format(item, i) for item in name['neg'][rand_list_neg] 
                    for i in os.listdir(self.base+'patch/'+item) if i.endswith('.jpg') and not i.endswith('_mask.jpg')]
            if self.pure_pos:
                if self.positive_picture_only:
                    self.neg = []
                self.pos = []
                for item in name['pos'][rand_list_pos]:
                    L = os.listdir(self.base+'patch/'+item)
                    for i in L:
                        if i.endswith('.jpg') and not i.endswith('_mask.jpg'):
                            if i.replace('.jpg', '_mask.jpg') in L:
                                self.pos.append(self.base+'patch/{}/{}'.format(item, i))
                            else:
                                self.neg.append(self.base+'patch/{}/{}'.format(item, i))
            else:
                self.pos = [self.base+'patch/{}/{}'.format(item, i) for item in name['pos'][rand_list_pos]
                    for i in os.listdir(self.base+'patch/'+item) if i.endswith('.jpg') and not i.endswith('_mask.jpg')]
        else:
            rand_list_neg = rand_list_neg[int(len(rand_list_neg)*self.train_rate):]
            rand_list_pos = rand_list_pos[int(len(rand_list_pos)*self.train_rate):]
            if self.whole:
                self.neg = list(map(lambda x:self.base+x+'.jpg',name['neg'][rand_list_neg]))
                self.pos = list(map(lambda x:self.base+x+'.jpg',name['pos'][rand_list_pos]))
            else:
                self.neg = [self.base+'patch/{}/{}'.format(item, i) for item in name['neg'][rand_list_neg] 
                        for i in os.listdir(self.base+'patch/'+item)]
                if self.pure_pos:
                    if self.positive_picture_only:
                        self.neg = []
                    self.pos = []
                    for item in name['pos'][rand_list_pos]:
                        L = os.listdir(self.base+'patch/'+item)
                        for i in L:
                            if i.endswith('.jpg') and not i.endswith('_mask.jpg'):
                                if i.replace('.jpg', '_mask.jpg') in L:
                                    self.pos.append(self.base+'patch/{}/{}'.format(item, i))
                                else:
                                    self.neg.append(self.base+'patch/{}/{}'.format(item, i))
                else:
                    self.pos = [self.base+'patch/{}/{}'.format(item, i) for item in name['pos'][rand_list_pos] 
                        for i in os.listdir(self.base+'patch/'+item) if i.endswith('.jpg') and not i.endswith('_mask.jpg')]
        self.pos_len = len(self.pos)
        self.neg_len = len(self.neg)
        print(self.pos_len,self.neg_len)
        self.transforms = transforms
        self.weights = [self.neg_len if i<self.pos_len else self.pos_len for i in range(self.pos_len+self.neg_len)]

    def __getitem__(self, idx):
        if idx<self.pos_len:
            name = self.pos[idx]
            targets = torch.ones(1).float()
        else:
            name = self.neg[idx-self.pos_len]
            targets = torch.zeros(1).float()
        if self.mode == 'test' and self.whole:
            return iter(SplitPatch(name,self.transforms,self.batch,self.patch_size,self.stride,self.pure_pos))
        else:
            inputs = [self.transforms(cv2.imread(name)[:,:,::-1]),name]
        return inputs, targets

    def __len__(self):
        return self.neg_len+self.pos_len

class SplitPatch(weak_SplitPatch):
    def __init__(self,name,transforms,batch,patch_size,stride,pure_pos):
        self.name = name
        self.pure_pos = pure_pos
        self.transforms = transforms
        self.IMG = cv2.imread(name)[:,:,::-1]
        self.mask = cv2.imread(name.replace('.jpg','_mask.jpg').replace('pos_','mask_'))
        w,h = self.IMG.shape[:2]
        super(SplitPatch,self).__init__(w,h,patch_size,stride,batch)
    
    def get_input(self,ws,we,hs,he):
        return self.transforms(self.IMG[ws:we, hs:he, :]).unsqueeze(0)
    
    def get_target(self,ws,we,hs,he):
        if self.mask is not None:
            return torch.Tensor([[self.mask[ws:we,hs:he].sum()>0]]).float() if self.pure_pos else torch.ones(1,1).float()
        else:
            return torch.zeros(1,1).float()

def dataloader(cfg, mode):
    if mode == 'train':
        transforms = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = Digestpath_fixed_list(cfg,mode,transforms)
        sampler = WeightedRandomSampler(dataset.weights,cfg['batch_size']*cfg['max_batch'],True)
        loader = torch.utils.data.DataLoader(dataset, 
            batch_size=cfg['batch_size'], sampler=sampler, num_workers=cfg['num_workers'])
    else:
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = Digestpath_fixed_list(cfg,mode,transforms)
        if cfg['whole_image']:
            loader = dataset
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'],shuffle=True,num_workers=cfg['num_workers'])
    if mode  == 'train':
        return loader, dataloader(cfg,'test')
    else:
        return loader
