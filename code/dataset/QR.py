import torchvision.transforms as T
import torch
import os
import cv2
import platform
import numpy as np
class QRloader(torch.utils.data.Dataset):
    def __init__(self,cfg,train,transform):
        super(QRloader, self).__init__()
        self.train = train
        if platform.system() == "Windows": idx=1
        else: idx=0
        if train:
            self.img_dir = cfg['direction'][idx]+'img/'
            self.label_dir = cfg['direction'][idx]+'label/'
            self.name_list = os.listdir(self.img_dir)
            self.len = len(self.name_list)
        else:
            self.img_dir = cfg['whole_image_dir'][idx]
            self.name_list = os.listdir(self.img_dir)
            self.len = len(self.name_list)
        self.image_shape = [tuple(x) for x in cfg['image_shape']]
        self.transform = transform
        self.readin_shape = tuple(cfg['readin_shape'])
        self.flod = torch.nn.Unfold((8, 8), padding=0, stride=8)

    def __getitem__(self,idx):
        img_dir = self.img_dir + self.name_list[idx]
        IMG = cv2.imread(img_dir)
        if self.train:
            inputs = self.transform(IMG).unsqueeze(0).float()
            inputs = self.flod(inputs).permute(0, 2, 1).contiguous().view(-1, 3, 8, 8)
            lab_dir = self.label_dir + self.name_list[idx][:-3] + 'npz'
            targets = torch.from_numpy(np.load(lab_dir)['data']).float().reshape(-1)
            targets[targets == 2] = 0.5
        else:
            targets = cv2.resize(IMG, self.readin_shape, interpolation=cv2.INTER_NEAREST)
            inputs = torch.Tensor().float()
            for box in self.image_shape:
                img = cv2.resize(targets, box, interpolation=cv2.INTER_NEAREST)
                img = self.transform(img).unsqueeze(0).float()
                inputs = torch.cat((inputs,self.flod(img).permute(0, 2, 1).contiguous().view(-1, 3, 8, 8)))
        return inputs,targets

    def __len__(self):
        return self.len


def dataloader(cfg, mode):
    train = mode=='train'
    transforms = T.Compose([
        T.ToTensor(),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = QRloader(cfg,train,transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['train_shuffle'], num_workers=cfg['num_workers'])
    if train:
        return loader,None
    else:
        return loader
