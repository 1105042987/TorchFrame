import os
import struct
import numpy as np
import torchvision.transforms as T
import torch


class MNIST(torch.utils.data.Dataset):
    def __init__(self,cfg,train):
        super(MNIST, self).__init__()
        path = cfg['path']
        self.train = train
        self.noise = cfg['noise_amplitude']
        self.label_noise = cfg['label_noise']
        self.image_shape = cfg['image_shape']
        if train:
            labels_path = os.path.join(path,'train-labels-idx1-ubyte')
            images_path = os.path.join(path,'train-images-idx3-ubyte')
        else:
            labels_path = os.path.join(path,'t10k-labels-idx1-ubyte')
            images_path = os.path.join(path,'t10k-images-idx3-ubyte')
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            labels = np.fromfile(lbpath,dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        pos_choose = labels==cfg['pos_num'][0]
        self.positive = torch.from_numpy(images[pos_choose]).reshape(-1,1,28,28).float()/255.0
        self.length = self.positive.shape[0]
        if not train:
            self.test_pos_rate = cfg['test_positive_rate']
            neg_choose = (labels==cfg['neg_num'][0])|(labels==cfg['neg_num'][1])
            self.negtive = torch.from_numpy(images[neg_choose])[:self.length,:].reshape(-1,1,28,28).float()/255.0


    def __getitem__(self, idx):
        noise = torch.randn(1,self.image_shape[0],self.image_shape[1])*self.noise
        label_noise = torch.rand(1)*self.label_noise
        if self.train:
            return [self.positive[idx, :, :, :], self.positive[idx, :, :, :]+noise], torch.Tensor([1-label_noise])
        else:
            if torch.rand(1) < self.test_pos_rate:
                return [self.positive[idx, :, :, :], self.positive[idx, :, :, :]+noise], torch.Tensor([1-label_noise])
            else:
                return [self.negtive[idx, :, :, :], self.negtive[idx, :, :, :]+noise], torch.Tensor([label_noise])

    def __len__(self):
        return self.length


def dataloader(cfg, mode):
    trian = mode=='train'
    dataset = MNIST(cfg,train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['train_shuffle'], num_workers=cfg['num_workers'])
    if train:
        dataset2 = MNIST(cfg, False)
        loader2 = torch.utils.data.DataLoader(dataset2, batch_size=40, shuffle=False, num_workers=cfg['num_workers'])
        # return loader, loader2
        return loader,None
    else:
        return loader

if __name__ == "__main__":
    cfg = {
        "file_name": "mnist",
        "path": "../../Dataset/MNIST/raw",
        "num_workers": 2,
        "train_shuffle": True,
        "pos_num": [1],
        "neg_num": [6, 7],
        "image_shape": [28, 28],
        "noise_amplitude": 0.05
    }
    dataset = MNIST(cfg, True)
    print(len(dataset))
