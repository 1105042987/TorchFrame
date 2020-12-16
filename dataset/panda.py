from __future__ import absolute_import
import os
import sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import torchvision.transforms as T
from torch.utils.data.sampler import WeightedRandomSampler
from docker.abstract_model import weak_SplitPatch
import pandas as pd
import torch
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, mode, transforms):
        super(Dataset, self).__init__()
        self.base = cfg['direction']
        self.mode = mode
        self.batch = cfg['batch_size']
        self.ISUP = cfg['ISUP']
        self.PatchSize = cfg.get('PatchSize', None)
        self.transforms = transforms

        self.imgBase = os.path.join(self.base,cfg['LoadName'][0])
        self.maskBase = os.path.join(self.base,cfg['LoadName'][1])
        df = pd.read_csv(os.path.join(self.base,cfg['LoadName'][2]))
        if self.PatchSize is None:
            PicSize = 10616
            idxList = df.PicCnt
        else:
            PicSize = len(df)
            idxList = df.index
        st,ed = cfg['TestInterval'][0]*PicSize, cfg['TestInterval'][1]*PicSize
        if mode == 'train':
            df = df[(idxList<st)|(idxList>=ed)]
        else:
            df = df[(idxList>=st)&(idxList<ed)]
        self.len = len(df)
        print(self.len)
        if cfg['LoadName'][2] == 'train.csv':
            self.name = list(df['image_id'])
            self.source = torch.from_numpy(df['data_provider'].apply(lambda x:int(x=='karolinska')).values)
            self.isup = torch.from_numpy(df['isup_grade'].values).long()
            self.gleason = torch.from_numpy(df['gleason_score'].replace('negative','0+0').\
                    apply(lambda x:pd.Series([int(x[0]),int(x[2])])).values).long()
        else:
            self.name = list(df['id'])
            self.source = torch.from_numpy(df['source'].apply(lambda x:int(x=='karolinska')).values)
            self.isup = torch.from_numpy(df['ori_i'].values).long()
            self.gleason = torch.from_numpy(df[['ori_g0','ori_g1']].values).long()
        

    def __getitem__(self,idx):
        if self.PatchSize is None:
            inputs = cv2.imread(os.path.join(self.imgBase,'{}.png'.format(self.name[idx])))[:,:,::-1]
        else:
            return iter(SplitPatch(os.path.join(self.imgBase,'{}.png'.format(self.name[idx])),
                    os.path.join(self.imgBase,'{}.png'.format(self.name[idx])),self.isup[idx],
                    self.transforms,self.batch,self.PatchSize,self.PatchSize//2,self.ISUP))
        if self.ISUP:
            targets = self.isup[idx]
        else:
            mask = cv2.imread(os.path.join(self.maskBase,'{}.png'.format(self.name[idx]))) # 必然存在
            cnts = []
            for i in range(6): cnts.append((mask==i).sum())
            # glea = torch.Tensor(cnts).sort()[1][[-1,-2]]
            targets = torch.Tensor(cnts).argmax()
        return self.transforms(inputs),targets

    def __len__(self):
        return self.len


class SplitPatch(weak_SplitPatch):
    def __init__(self,imgdir,maskdir,whole_isup,transforms,batch,patch_size,stride,isup_only):
        self.isup_only = isup_only
        self.whole_isup = whole_isup
        self.transforms = transforms
        self.IMG = cv2.imread(imgdir)[:,:,::-1]
        self.mask = cv2.imread(maskdir) if os.path.exists(maskdir) else None
        w,h = self.IMG.shape[:2]
        # padding
        ud = patch_size - w if patch_size > w else 0
        lr = patch_size - h if patch_size > h else 0
        if ud + lr!=0:
            self.IMG = cv2.copyMakeBorder(self.IMG,ud//2,ud-ud//2,lr//2,lr-lr//2,cv2.BORDER_CONSTANT)
            self.mask = cv2.copyMakeBorder(self.mask,ud//2,ud-ud//2,lr//2,lr-lr//2,cv2.BORDER_CONSTANT)
        super(SplitPatch,self).__init__(w,h,patch_size,stride,batch)
    
    def get_input(self,ws,we,hs,he):
        return self.transforms(self.IMG[ws:we, hs:he, :])
    
    def get_target(self,ws,we,hs,he):
        if self.isup_only:
            return self.whole_isup
        else:
            mask = self.mask[ws:we, hs:he, :]
            cnts = []
            for i in range(6): cnts.append((mask==i).sum())
            return torch.Tensor(cnts).argmax()


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
    else:
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    dataset = Dataset(cfg,mode,transforms)
    if cfg.get('PatchSize',None) is not None:
        return dataset
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], 
            shuffle=cfg['shuffle'], num_workers=cfg['num_workers'])
    return loader
        


def __generate_png(base = '/remote-home/source/DATA/PANDA/', choose=[1,2]):
    from libtiff import TIFF
    import skimage
    from tqdm import tqdm
    fp = os.path.join(base,'train.csv')
    name = pd.read_csv(fp)['image_id']
    root = os.path.join(base,'train_images')
    mask = os.path.join(base,'train_label_masks')
    tar = [
        os.path.join(base,'train_png_h'),
        os.path.join(base,'train_png_m'),
        os.path.join(base,'train_png_s'),
        os.path.join(base,'train_mask_h'),
        os.path.join(base,'train_mask_m'),
        os.path.join(base,'train_mask_s'),
    ]
    for t in tar:
        if not os.path.exists(t): os.makedirs(t)
    for n in tqdm(name,ascii=True):
        imgdir = os.path.join(root,n+'.tiff')
        imgs = skimage.io.MultiImage(imgdir)
        maskdir = os.path.join(mask,n+'_mask.tiff')
        masks = skimage.io.MultiImage(maskdir)
        for idx in choose:
            cv2.imwrite(os.path.join(tar[idx],n+'.png'),imgs[idx])
            if os.path.exists(maskdir):
                cv2.imwrite(os.path.join(tar[idx+3],n+'.png'),masks[idx])

        # tif = TIFF.open(imgdir,mode='r')
        # cnt=0
        # for img in list(tif.iter_images()):
        #     if cnt not in skip: 
        #         cv2.imwrite(os.path.join(tar[cnt],n.replace('tiff','png')),img)
        #     cnt+=1

def __generate_patch(base = '/remote-home/source/DATA/PANDA/', FROM = 's', patch_size=256, stride=128):
    from tqdm import tqdm
    import numpy as np
    # TAR = os.path.join('/root/qsf','train_patch_{}'.format(FROM))
    TAR = os.path.join(base,'train_patch_{}'.format(FROM))
    MASK = os.path.join(base,'train_mask_{}'.format(FROM))
    IMG = os.path.join(base,'train_png_{}'.format(FROM))
    fp = os.path.join(base,'train.csv')
    data = pd.read_csv(fp).values
    cnt = 0
    PicCnt = 0
    if not os.path.exists(TAR):
        os.makedirs(os.path.join(TAR,'img'))
        os.makedirs(os.path.join(TAR,'mask'))
    if not os.path.exists(os.path.join(TAR,'label.csv')):
        jump=0
        with open(os.path.join(TAR,'label.csv'),'w') as f:
            f.write('id,PicCnt,source,ori_i,ori_g0,ori_g1\n')
    else:
        jump = len(pd.read_csv(os.path.join(TAR,'label.csv')))

    for d in tqdm(data,ascii=True):
        n,s,isup,glea = d
        if '+' in glea:
            g0,g1 = int(glea[0]),int(glea[2])
        else:
            g0,g1 = 0,0
        im = cv2.imread(os.path.join(IMG,n+'.png'))
        if os.path.exists(os.path.join(MASK,n+'.png')):
            ma = cv2.imread(os.path.join(MASK,n+'.png'))[:,:,0]
        else:
            ma = None
        w,h = im.shape[:2]
        ud = patch_size - w if patch_size > w else 0
        lr = patch_size - h if patch_size > h else 0
        if ud + lr != 0:
            im = cv2.copyMakeBorder(im,ud//2,ud-ud//2,lr//2,lr-lr//2,cv2.BORDER_CONSTANT,value=255)
            ma = cv2.copyMakeBorder(ma,ud//2,ud-ud//2,lr//2,lr-lr//2,cv2.BORDER_CONSTANT,value=0)
        WS, WE = weak_SplitPatch.StartEndSplit(w, patch_size, stride)
        HS, HE = weak_SplitPatch.StartEndSplit(h, patch_size, stride)
        im[im>240] = 255
        for ws, we in zip(WS,WE):
            for hs, he in zip(HS,HE):
                patch = im[ws:we,hs:he,:]
                if patch.mean() > 250:
                    continue
                if cnt < jump:
                    cnt+=1
                    continue
                if ma is None:
                    mask = np.zeros((patch_size,patch_size))
                else:
                    mask = ma[ws:we,hs:he]
                cv2.imwrite(os.path.join(TAR,'img','{}.png'.format(cnt)),patch)
                cv2.imwrite(os.path.join(TAR,'mask','{}.png'.format(cnt)),mask)
                with open(os.path.join(TAR,'label.csv'),'a') as f:
                    f.write('{},{},{},{},{},{}\n'.format(cnt,PicCnt,s,isup,g0,g1))
                cnt+=1
        PicCnt+=1


if __name__ == "__main__":
    # __generate_png()
    __generate_patch(FROM = 'm', patch_size=512, stride=256)
    
    pass
