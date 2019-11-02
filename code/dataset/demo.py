import torchvision.transforms as T
import torch

class Digestpath(torch.utils.data.Dataset):
    def __init__(self):
        super(Digestpath, self).__init__()

    def __getitem__(self,idx):
        inputs = [torch.ones((1, 25, 25)), torch.ones((1, 25, 25))]
        targets = torch.ones(1)
        return inputs,targets

    def __len__(self):
        return 20


def dataloader(cfg,mode):
    trian = mode=='train'
    transforms = T.Compose([
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = Digestpath()
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['train_shuffle'], num_workers=cfg['num_workers'])
    if train:
        return loader,None
    else:
        return loader
