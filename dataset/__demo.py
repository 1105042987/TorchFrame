import torchvision.transforms as T
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        super(Dataset, self).__init__()

    def __getitem__(self,idx):
        inputs = None
        targets = None
        return inputs,targets

    def __len__(self):
        return 20


def dataloader(cfg, mode):
    transforms = T.Compose([
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = Dataset(cfg,mode)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['shuffle'], num_workers=cfg['num_workers'])
    return loader
