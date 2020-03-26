import torchvision.transforms as T
import torch
import os
import cv2
import random
import numpy as np

class BBoxRandomCrop:
    def __init__(self, size, numBBoxMax = 100,padding=None, pad_if_needed=False, fill=0, min_ratio=0.1):
        '''
        size(height, width)
        padding(left, right, top, bottom)
        '''
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.min_ratio = min_ratio
        self.numBBoxMax = numBBoxMax

    def __call__(self,img,bbox):
        shape = list(img.shape[:2])
        pad = [0,0]
        if self.padding is not None:
            pad[0] += self.padding[0]
            pad[1] += self.padding[2]
            shape[0] += self.padding[0]+self.padding[1]
            shape[1] += self.padding[2]+self.padding[3]
        if self.pad_if_needed:
            height = self.size[0]-shape[0] if self.size[0] > shape[0] else 0
            width = self.size[1]-shape[1] if self.size[1] > shape[1] else 0
            pad[0] += height
            pad[1] += width
            shape[0] += height*2
            shape[1] += width*2

        hs = random.randint(0, shape[0]-self.size[0])
        ws = random.randint(0, shape[1]-self.size[1])
        imghs = max(hs-pad[0], 0)
        imgws = max(ws-pad[1], 0)
        tmphs = max(pad[0]-hs, 0)
        tmpws = max(pad[1]-ws, 0)
        lenh = min(pad[0]+img.shape[0]-hs, self.size[0]-tmphs)
        lenw = min(pad[1]+img.shape[1]-ws, self.size[1]-tmpws)
        tmp = (np.ones((self.size[0], self.size[1], 3))*self.fill).astype(np.uint8)
        tmp[tmphs:tmphs+lenh, tmpws:tmpws+lenw] = img[imghs:imghs+lenh, imgws:imgws+lenw, :]

        clean_bbox = np.zeros((self.numBBoxMax,4))
        max_size = (bbox[:,1,:]-bbox[:,0,:]).prod(1).max()
        num=0
        for item in bbox:
            #item: scol,srow,ecol,erow
            it = [min(max(item[0,1]+pad[0]-hs, 0), self.size[0]-1), min(max(item[0,0]+pad[1]-ws, 0),self.size[1]-1),
                  min(max(item[1,1]+pad[0]-hs, 0), self.size[0]-1), min(max(item[1,0]+pad[1]-ws, 0),self.size[1]-1)]
            #it: srow,scol,erow,ecol
            if (it[2]-it[0])*(it[3]-it[1])>self.min_ratio*max_size:
                clean_bbox[num, :] = np.array(it)
                num+=1
        return tmp,(clean_bbox,num)


class QRloader(torch.utils.data.Dataset):
    def __init__(self,cfg,train,transform):
        super(QRloader, self).__init__()
        self.train = train
        self.basePath = cfg['direction']
        self.img_list = os.listdir(self.basePath)
        self.img_list.remove('bbox')
        self.input_shape = (cfg['input_shape'][1],cfg['input_shape'][0])
        self.numBBoxMax = cfg['random_crop']['numBBoxMax']

        self.len = len(self.img_list)
        self.crop = BBoxRandomCrop(**cfg['random_crop'])
        self.transform = transform

    def __getitem__(self,idx):
        img_dir = os.path.join(self.basePath,self.img_list[idx])
        lab_dir = os.path.join(self.basePath,'bbox',self.img_list[idx].replace('.jpg','.npz'))
        IMG = cv2.resize(cv2.imread(img_dir), self.input_shape, interpolation=cv2.INTER_NEAREST)       
        bbox = np.load(lab_dir)['data'] if os.path.exists(lab_dir) else None
        if self.train:
            IMG,LAB = self.crop(IMG,bbox)
        else:
            if bbox is None:
                return self.transform(IMG),None
            clean_bbox = np.zeros((self.numBBoxMax,4))
            num=0
            for item in bbox:
                clean_bbox[num, :] = np.array([item[0,1],item[0,0],item[1,1],item[1,0]])
                num+=1
            LAB = (clean_bbox,num)
        # return IMG,LAB                            # just for test
        return self.transform(IMG),LAB

    def __len__(self):
        return self.len


def dataloader(cfg, mode):
    train = mode=='train'
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = QRloader(cfg,train,transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers'])
    return loader


if __name__ == "__main__":
    import time
    dataset={
        "num_workers": 2,
        "input_shape":[504,672],
        "random_crop":{
            "size":[504,672],
            "padding":[128,128,128,128],
            "numBBoxMax":100
        },
        "direction": r"D:\Data\QRdata\part\mixup",
        "batch_size": 1,
        "shuffle": True
    }
    load = dataloader(dataset,'train')
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    s = time.time()
    for item in load:
        tmp,clean = item
        tmp=np.array(tmp[0]).astype(np.uint8)
        transforms(tmp)
        print((time.time()-s)*1000)
        bbox = clean[0][0]
        num = clean[1][0]
        for item in bbox[:num]:
            cv2.rectangle(tmp,(int(item[1]),int(item[0])),(int(item[3]),int(item[2])),(0,255,0),1)
        cv2.imshow('img',tmp)
        k=cv2.waitKey(0)
        if k==ord('q'): break
        s = time.time()
        
