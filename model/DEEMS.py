from __future__ import absolute_import
import sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import cv2
from docker.abstract_model import weak_evaluate, weak_loss

class basic_rnn(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layer):
        super(basic_rnn, self).__init__()
        self.rnn = nn.RNN(input_dim,hidden_dim,num_layer)
        self.linear = nn.Linear(hidden_dim,1,bias=True)
    def forward(self,x):
        out,hid = self.rnn(x)
        return self.linear(out).reshape(-1)

class attention(nn.Module):
    def __init__(self,hidden_dim):
        super(attention, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2*hidden_dim, 1, bias=True),
            nn.Tanh(),
            nn.Softmax(dim=1),
        )
        self.dim = hidden_dim
    def forward(self,data,key):
        data_ = data.unsqueeze(1).repeat([1,key.shape[1],1])
        attn_in = torch.cat([data_,key],2)
        attn = self.layer(attn_in).repeat([1,1,key.shape[2]])
        return (attn*key).sum(1).reshape(-1,self.dim)

class net(nn.Module):
    def __init__(self, item_num, user_num, hidden_units_u, hidden_units_i, position_len, strengthen_method):
        super(net, self).__init__()
        user_num, item_num = int(item_num), int(user_num)
        # part1
        self.user1_emb = nn.Embedding(user_num, hidden_units_u, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.item1_emb = nn.Embedding(item_num, hidden_units_i, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.user1_bias = Parameter(torch.Tensor(user_num))
        self.user1_bias.data.uniform_(-1, 1)
        self.item1_bias = Parameter(torch.Tensor(item_num))
        self.item1_bias.data.uniform_(-1, 1)
        self.u_pos_emb = nn.Embedding(position_len, hidden_units_i, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        # part 2
        self.user2_emb = nn.Embedding(user_num, hidden_units_u, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.item2_emb = nn.Embedding(item_num, hidden_units_i, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        self.user2_bias = Parameter(torch.Tensor(user_num))
        self.user2_bias.data.uniform_(-1, 1)
        self.item2_bias = Parameter(torch.Tensor(item_num))
        self.item2_bias.data.uniform_(-1, 1)
        self.i_pos_emb = nn.Embedding(position_len, hidden_units_u, padding_idx=None,
                           max_norm=None, norm_type=2, scale_grad_by_freq=False, sparse=False)
        # public
        self.strengthen_method = strengthen_method
        if self.strengthen_method == 'RNN':
            self.RNN_u = basic_rnn(hidden_units_u, hidden_units_u,1)
            self.RNN_i = basic_rnn(hidden_units_i, hidden_units_i,1)
        elif self.strengthen_method == 'ATTN':
            self.attn_u = attention(hidden_units_u)
            self.attn_i = attention(hidden_units_i)
        else:
            raise('Strengthen method name not exist')
        # 上面这些 "self.RNN_u","self.RNN_i","self.attn_u","self.attn_i"在原先的代码中有点奇怪,我不太确定他们是哪些共用一组网络参数.

    def forward(self, xs):
        user1, item1, user2, item2, u_his, u_pos, u_his_mask, i_his, i_pos, i_his_mask = xs
        user1, item1, user2, item2 = user1.reshape(-1), item1.reshape(-1), user2.reshape(-1), item2.reshape(-1)
        # part 1: user-aspect
        u1s_emb = self.user1_emb(user1)
        i1s_emb = self.item1_emb(item1)
        u1s_b = self.user1_bias.gather(0, user1)
        i1s_b = self.item1_bias.gather(0, item1)
        u1r_emb = self.user2_emb(user1).detach()
        i1r_emb = self.item2_emb(item1).detach()
        u1r_b = self.user2_bias.gather(0, user1).detach()
        i1r_b = self.item2_bias.gather(0, item1).detach()

        u_hiss_emb = self.item1_emb(u_his)
        i_hisr_emb = self.user2_emb(i_his).detach()
        u_hiss_emb[u_his_mask==0,...]=0
        i_hisr_emb[i_his_mask==0,...]=0

        if self.strengthen_method == 'ATTN':
            u_poss_emb = self.u_pos_emb(u_pos)
            i_posr_emb = self.i_pos_emb(i_pos).detach()
            u_poss_emb[u_his_mask==0,...]=0
            i_posr_emb[i_his_mask==0,...]=0
            u_ds_emb = self.attn_u(i1s_emb, u_hiss_emb+u_poss_emb)
            i_dr_emb = self.attn_i(u1r_emb, i_hisr_emb+i_posr_emb)
            logits_us = u1s_b + i1s_b + ((u_ds_emb+i1s_emb)*u1s_emb).sum(1)
            logits_ir = u1r_b + i1r_b + ((i_dr_emb+i1r_emb)*u1r_emb).sum(1)
        else:
            u_nn_l1s = self.RNN_u(u_hiss_emb)
            i_nn_l1r = self.RNN_i(i_hisr_emb)
            logits_us = u_nn_l1s + u1s_b + i1s_b + (i1s_emb*u1s_emb).sum(1)
            logits_ir = i_nn_l1r + u1r_b + i1r_b + (i1r_emb*u1r_emb).sum(1)


        # part 2: item-aspect
        u2s_emb = self.user2_emb(user2)
        i2s_emb = self.item2_emb(item2)
        u2s_b = self.user2_bias.gather(0, user2)
        i2s_b = self.item2_bias.gather(0, item2)
        u2r_emb = self.user1_emb(user2).detach()
        i2r_emb = self.item1_emb(item2).detach()
        u2r_b = self.user1_bias.gather(0, user2).detach()
        i2r_b = self.item1_bias.gather(0, item2).detach()

        u_hisr_emb = self.item1_emb(u_his).detach() # 这里原代码是从item2_emd提取，但是我觉得有点奇怪就给改了
        i_hiss_emb = self.user2_emb(i_his)
        u_hisr_emb[u_his_mask==0,...]=0
        i_hiss_emb[i_his_mask==0,...]=0

        if self.strengthen_method == 'ATTN':
            u_posr_emb = self.u_pos_emb(u_pos).detach()
            i_poss_emb = self.i_pos_emb(i_pos)
            u_posr_emb[u_his_mask==0,...]=0
            i_poss_emb[i_his_mask==0,...]=0
            u_dr_emb = self.attn_u(i1r_emb, u_hisr_emb+u_posr_emb)
            i_ds_emb = self.attn_i(u1s_emb, i_hiss_emb+i_poss_emb)
            logits_ur = u2r_b + i2r_b + ((u_dr_emb+i2r_emb)*u2r_emb).sum(1)
            logits_is = u2s_b + i2s_b + ((i_ds_emb+i2s_emb)*u2s_emb).sum(1)
        else:
            u_nn_l1r = self.RNN_u(u_hisr_emb)
            i_nn_l1s = self.RNN_i(i_hiss_emb)
            logits_ur = u_nn_l1r + u2r_b + i2r_b + (i2r_emb*u2r_emb).sum(1)
            logits_is = i_nn_l1s + u2s_b + i2s_b + (i2s_emb*u2s_emb).sum(1)
        
        loss_reg1 = (u1s_emb**2).sum() + (i1s_emb**2).sum() + (u_hiss_emb**2).sum()
        loss_reg2 = (u2s_emb**2).sum() + (i2s_emb**2).sum() + (i_hiss_emb**2).sum()
        return [logits_us, logits_ir, logits_ur, logits_is, loss_reg1, loss_reg2]

class loss(weak_loss):
    def __init__(self,reg,hedge1,hedge2):
        super(loss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()  # 内含softmax
        self.sigmoid = nn.Sigmoid()
        self.reg = reg
        self.hedge1 = hedge1
        self.hedge2 = hedge2

    def get_loss(self, pre, tar):
        logits_us, logits_ir, logits_ur, logits_is, loss_reg1, loss_reg2 = pre
        pred_us, pred_ir = self.sigmoid(logits_us), self.sigmoid(logits_ir)
        pred_ur, pred_is =self.sigmoid(logits_ur), self.sigmoid(logits_is)
        label1 , label2 = [x.float() for x in tar]

        logits = ((logits_us + logits_is)/2)
        loss_r = self.loss(logits.reshape(-1, 1), label1).mean()
        loss_r1 = self.loss(logits_us.reshape(-1, 1), label1).mean()
        reward1 = -( pred_ir*torch.log(pred_us+1e-6) + (1-pred_ir)*torch.log(1-pred_us+1e-6) ).mean()
        loss_hedge1 = ((1-label1).float()*reward1).mean()
        loss1 = loss_r1 + self.reg*loss_reg1 + self.hedge1*loss_hedge1 

        loss_r2 = self.loss(logits_is.reshape(-1, 1), label2).mean()
        reward2 = -(pred_is+torch.log(pred_ur+1e-6) + (1-pred_is)*torch.log(1-pred_ur+1e-6)).mean()
        loss_hedge2 = ((1-label2).float()*reward2).mean()
        loss2 = loss_r2 + self.reg*loss_reg2 + self.hedge2*loss_hedge2
        return loss1+loss2,{"loss1":loss1,"loss2":loss2}

class evaluate(weak_evaluate):
    def __init__(self, result_dir):
        super(evaluate, self).__init__(result_dir)

    def get_eval(self, inputs, preds, targets):
        return {}

    def visualize(self, inputs, preds, targets, _eval):
        return None

    def save(self, inputs, preds, targets, _eval):
        pass

if __name__ == "__main__":
    net_param={
        "item_num": 20,
        "user_num": 20,
        "hidden_units_u": 20,
        "hidden_units_i": 20,
        "position_len": 20,
        "strengthen_method": "ATTN"
    }
    loss_param={
        "reg": 1,
        "hedge1": 1,
        "hedge2": 1
    }
    n = net(**net_param)
    l = loss(**loss_param)
    user1, item1 = torch.zeros(2).long(), torch.zeros(2).long()
    user2, item2 = torch.zeros(2).long(), torch.zeros(2).long()
    u_his, u_pos, u_his_mask = torch.zeros(2,5).long(), torch.zeros(2,5).long(), torch.zeros(2,5).long()
    i_his, i_pos, i_his_mask = torch.zeros(2,5).long(), torch.zeros(2,5).long(), torch.zeros(2,5).long()
    i = [user1, item1, user2, item2, u_his, u_pos, u_his_mask, i_his, i_pos, i_his_mask]
    tar = [torch.zeros(2).long(), torch.zeros(2).long()]

    pre = n(i)
    loss,dic = l(pre,tar)
    loss.backward()
