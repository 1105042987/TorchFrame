import torch 
import torchvision.transforms as T
import numpy as np
import random
import pickle

random.seed(1234)

class DataInput(torch.utils.data.Dataset):
    def __init__(self, cfg, train):
        super(DataInput, self).__init__()
        with open(cfg['path']+'Clothing_dataset_-2_new.pkl', 'rb') as f:
            train_set = pickle.load(f)
            test_set = pickle.load(f)
            u_his_list = pickle.load(f)
            i_his_list = pickle.load(f)
            user_count, item_count, example_count = pickle.load(f)
            print(user_count, item_count)
        if train: self.data = train_set
        else: self.data = test_set
        self.u_his_list = u_his_list
        self.i_his_list = i_his_list
        self.user_count = user_count
        self.item_count = item_count
        self.trunc_len = cfg['trunc_len']
        self.neg_rate = cfg['neg_rate']
        self.len = len(self.data)

    def padding(self,list_):
        temp = torch.Tensor(list_).long()
        zero = torch.zeros(self.trunc_len).long()
        zero[:len(temp)]=temp
        return zero

    def __getitem__(self,idx):
        t = self.data[idx]
        uid_1 = torch.Tensor([t[0]]).long()
        iid_1 = torch.Tensor([t[1]]).long()
        label_1 = torch.Tensor([t[2]]).long()
        u_his = []
        for i in self.u_his_list[t[0]]:
            if i == t[1]: break  # 为什么不是continue
            u_his.append(i)
        u_his_l = len(u_his)
        if u_his_l > self.trunc_len: u_his_l = self.trunc_len
        u_his_mask = torch.zeros(self.trunc_len).long()
        u_his_mask[:u_his_l]=1
        u_his = self.padding(u_his[-self.trunc_len:])
        u_pos = self.padding([u_his_l - j - 1 for j in range(u_his_l)])

        i_his_all = self.i_his_list[t[1]]
        i_his = []
        for u in i_his_all:
            if u == t[0]: break
            i_his.append(u)
        i_his_l = len(i_his)
        if i_his_l > self.trunc_len: i_his_l = self.trunc_len
        i_his_mask = torch.zeros(self.trunc_len).long()
        i_his_mask[:i_his_l] = 1
        i_his = self.padding(i_his[-self.trunc_len:])
        i_pos = self.padding([i_his_l - j - 1 for j in range(i_his_l)])

        #generate negative sample
        if torch.rand(1)<self.neg_rate:
            while iid_1 == t[1]: iid_1 = torch.randint(self.item_count, (1,)).long()
            label_1 = torch.Tensor([0]).long()

        #sample for user RNN
        uid_2 = torch.Tensor([t[0]]).long()
        iid_2 = torch.Tensor([t[1]]).long()
        label_2 = torch.Tensor([t[2]]).long()
        if torch.rand(1) < self.neg_rate:
            while uid_2 == t[0]: uid_2 = torch.randint(self.item_count, (1,)).long()
            label_2 = torch.Tensor([0]).long()
        return [uid_1, iid_1, uid_2, iid_2, u_his, u_pos, u_his_mask, i_his, i_pos, i_his_mask], [label_1, label_2]

    def __len__(self):
        return self.len
    
def dataloader(cfg, mode):
    train = mode=='train'
    dataset = DataInput(cfg,train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=cfg['train_shuffle'], num_workers=cfg['num_workers'])
    return loader,None
