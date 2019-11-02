from __future__ import absolute_import
import sys
sys.path.append('..')
import cv2
import torch 
import numpy as np
import torch.nn as nn
from docker.abstract_model import weak_evaluate, weak_loss

class net(nn.Module):
    def __init__(self,**args):
        super(net,self).__init__()
        self.G_net = generator(**args)

    def forward(self,xs):
        return self.G_net(xs)


class loss(weak_loss):
    def __init__(self,D_net,D_opt,gan_k,loss_weight):
        super(loss, self).__init__()
        self.D_net = discriminator(**D_net)
        self.D_opt = torch.optim.Adam(self.D_net.parameters(), **D_opt)

        self.w = loss_weight
        # MSE loss between Generated image and original image
        # self.Lg = nn.BCEWithLogitsLoss()
        self.Lg = nn.L1Loss()
        # Discriminartor Loss
        self.Ld_pos = nn.BCEWithLogitsLoss()    
        self.Ld_neg = nn.BCEWithLogitsLoss()
        # Mixed loss
        self.Lgd = nn.BCEWithLogitsLoss()
        self.gan_k = gan_k
        self.train_d = True

    def state_dict(self):
        return {'net':self.D_net.state_dict(),'opt':self.D_opt.state_dict()}
    
    def to(self,dev):
        self.D_net = self.D_net.to(dev)
        return self
    
    def load_state_dict(self,dic):
        self.D_net.load_state_dict(dic['net'])
        self.D_opt.load_state_dict(dic['opt'])

    def get_loss(self, preds, targets):
        loss_d = 0  
        x_ori, x_gen = preds
        loss_g = self.Lg(x_ori, x_gen)

        detach = [x.detach() for x in preds]

        for _ in range(self.gan_k):
            self.D_opt.zero_grad()
            d_ori, d_gen = self.D_net(detach)

            label_ori = torch.ones_like(d_ori)
            label_gen = torch.zeros_like(d_gen)
            loss = self.Ld_pos(d_ori, label_ori) + self.Ld_neg(d_gen, label_gen)
            loss_d += loss
            if self.train_d:
                loss.backward()
                self.D_opt.step()
        loss_d /= self.gan_k
        if loss_d > 2: self.train_d = True
        if loss_d <=0.5: self.train_d = False

        d_ori, d_gen = self.D_net(preds)
        loss_g_d = self.Lgd(d_gen,label_ori)
        preds.append(d_ori)
        preds.append(d_gen)

        return self.w[0]*loss_g+self.w[1]*loss_g_d, {'D_loss':loss_d,'BCEx':loss_g,'G_loss':loss_g_d}



class evaluate(weak_evaluate):
    def __init__(self, result_dir, novelty_threshold):
        super(evaluate, self).__init__(result_dir)
        self.threshold = novelty_threshold

    def get_eval(self, inputs, preds, targets):
        x_ori, x_gen, d_ori, d_gen = preds
        val_D = ((d_ori>self.threshold)==(targets>0.5)).sum().float()/len(targets)
        val_D_G = ((d_gen>self.threshold)==(targets>0.5)).sum().float()/len(targets)
        return {"Pure_D_acc":val_D,"D&G_acc":val_D_G}

    def visualize(self, inputs, preds, targets, _eval):
        for x_ori, x_gen, d_ori, d_gen in zip(*preds):
            cv2.imshow('ori',np.array(x_ori.permute(1,2,0).cpu().detach()))
            cv2.imshow('gen',np.array(x_gen.permute(1,2,0).cpu().detach()))
            print('\nori:',float(nn.Sigmoid()(d_ori)),' gen:',float(nn.Sigmoid()(d_gen)))
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        return key

    def save(self, inputs, preds, targets, _eval):
        pass


class generator(nn.Module):
    def __init__(self, kernel_size, input_dim, growth_dim, nStride):
        super(generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, growth_dim, (kernel_size,
                                              kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim),
            nn.LeakyReLU(0.1),
            nn.Conv2d(growth_dim, growth_dim*2, (kernel_size,
                                                 kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(growth_dim*2, growth_dim*4, (kernel_size,
                                                   kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*4),
            nn.LeakyReLU(0.1),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(growth_dim*4, growth_dim*2,
                               (kernel_size, kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                growth_dim*2, growth_dim, (kernel_size, kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(
                growth_dim, input_dim, (kernel_size, kernel_size), stride=nStride, bias=True),
            nn.Tanh(),
        )

    def forward(self, xs):
        x_ori, x_noi = xs
        z_noi = self.encoder(x_noi)
        x_gen = self.decoder(z_noi)
        return [x_ori, x_gen]


class discriminator(nn.Module):
    def __init__(self, input_shape, kernel_size, input_dim, growth_dim, nStride):
        super(discriminator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, growth_dim, (kernel_size,
                                              kernel_size), stride=nStride, bias=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(growth_dim, growth_dim*2, (kernel_size,
                                                 kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(growth_dim*2, growth_dim*4, (kernel_size,
                                                   kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*4),
            nn.LeakyReLU(0.1),
            nn.Conv2d(growth_dim*4, growth_dim*8, (kernel_size,
                                                   kernel_size), stride=nStride, bias=True),
            nn.BatchNorm2d(growth_dim*8),
            nn.LeakyReLU(0.1),
        )
        length = (input_shape[0]-(kernel_size-1)*4) * \
            (input_shape[1]-(kernel_size-1)*4)*growth_dim*8
        if length <= 0:
            raise('Error input shape')
        self.classifier = nn.Sequential(
            nn.Linear(length, 1),
        )

    def forward(self, xs):
        x_ori, x_gen = xs
        batch = x_ori.shape[0]
        z_ori = self.encoder(x_ori)
        z_gen = self.encoder(x_gen)
        d_ori = self.classifier(z_ori.reshape(batch, -1))
        d_gen = self.classifier(z_gen.reshape(batch, -1))
        return [d_ori, d_gen]

if __name__ == "__main__":
    dic = {
        "input_shape": [25, 25],
        "kernel_size": 5,
        "input_dim": 1,
        "growth_dim": 16,
        "nStride": 1
    }
    n2 = discriminator(**dic)
    dic = {
        "kernel_size": 5,
        "input_dim": 1,
        "growth_dim": 32,
        "nStride": 1
    }
    n1 = net(**dic)
    i = torch.ones((4,1,25,25))
    o1 = n1([i,i])
    o2 = n2(o1)
    for idx,k in enumerate(o1):
        print(idx,'\t',k.shape)
    for idx, k in enumerate(o2):
        print(idx, '\t', k.shape)
