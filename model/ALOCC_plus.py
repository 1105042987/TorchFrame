from __future__ import absolute_import
import sys
sys.path.append('..')
import cv2
import torch 
import numpy as np
import torch.nn as nn
from docker.abstract_model import weak_evaluate,weak_loss

loss_d = 0
class net(nn.Module):
    def __init__(self,G_net,D_net,D_opt,gan_k):
        super(net,self).__init__()
        self.G_net = generator(**G_net)
        self.D_net = discriminator(**D_net)
        self.D_opt = torch.optim.Adam(self.D_net.parameters(), **D_opt)
        # Discriminartor Loss
        self.Ld_pos = nn.BCEWithLogitsLoss()
        self.Ld_neg = nn.BCEWithLogitsLoss()
        self.gan_k = gan_k
        self.train_d = True

    def forward(self,xs):
        preds = self.G_net(xs)
        # train Discriminator
        global loss_d = 0
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
        if loss_d > 2:
            self.train_d = True
        if loss_d <= 0.5:
            self.train_d = False
        # use Discriminator
        d_ori, d_gen = self.D_net(preds)
        preds.append(d_ori)
        preds.append(d_gen)
        return preds

    def state_dict(self):
        return {
            'G_net':self.G_net.state_dict(),
            'D_net':self.D_net.state_dict(),
            'D_opt':self.D_opt.state_dict(),
        }

    def to(self,dev):
        self.D_net = self.D_net.to(dev)
        self.G_net = self.G_net.to(dev)
        return self

    def load_state_dict(self,dic):
        self.G_net.load_state_dict(dic['G_net'])
        self.D_net.load_state_dict(dic['D_net'])
        self.D_opt.load_state_dict(dic['D_opt'])
        

class loss(weak_loss):
    def __init__(self,gan_k,loss_weight):
        super(loss, self).__init__()
        self.w = loss_weight
        # MAE loss between Generated image and original image
        self.Lg = nn.L1Loss()
        # MSE loss between Encoder1 and Encoder2
        self.Le = nn.MSELoss()
        # Mixed loss
        self.Lgd = nn.BCEWithLogitsLoss()
        
    def get_loss(self, preds, targets):
        global loss_d
        x_ori, x_gen, z_noi, z_gen, d_ori, d_gen = preds
        loss_g = self.Lg(x_ori, x_gen)
        loss_e = self.Le(z_noi, z_gen)
        
        label_ori = torch.ones_like(d_ori)
        loss_g_d = self.Lgd(d_gen,label_ori)
        
        total_loss = self.w[0]*loss_g + self.w[1]*loss_e + self.w[2]*loss_g_d
        return total_loss, {'Dis': loss_d, 'MAEx': loss_g, 'MSEz': loss_e, 'Dis(Gen)': loss_g_d}



class evaluate(weak_evaluate):
    def __init__(self, novelty_threshold, weights):
        super(evaluate, self).__init__()
        self.threshold = novelty_threshold
        self.weights = weights

    def get_eval(self, inputs, preds, targets):
        x_ori, x_gen, z_noi, z_gen, d_ori, d_gen = preds
        pix = x_ori.shape[-1]*x_ori.shape[-2]
        self.loss = torch.sum(torch.abs(x_ori-x_gen),[1,2,3])*self.weights[0]/pix+\
                torch.sum((z_noi-z_gen)**2,[1,2,3])*self.weights[1]/pix
        self.show_label = (self.loss < self.threshold)
        self.true = (self.loss < self.threshold) == (targets.reshape(-1) > 0.5)
        val = float(((self.loss < self.threshold) == (targets.reshape(-1)>0.5)).sum())/len(targets)
        return {'acc':val}
        # val_D = ((d_ori>self.threshold)==(targets>0.5)).sum()/len(targets)
        # val_D_G = ((d_gen>self.threshold)==(targets>0.5)).sum()/len(targets)
        # return {"Pure_D_acc":val_D,"D&G_acc":val_D_G}

    def visualize(self, inputs, preds, targets, _eval):
        key = ''
        for x_ori, x_gen, z_noi, z_gen, d_ori, d_gen, label, loss, true in zip(*preds,self.show_label,self.loss,self.true):
            cv2.imshow('ori', np.array(x_ori.permute(1, 2, 0).cpu().detach()))
            cv2.imshow('gen', np.array(x_gen.permute(1, 2, 0).cpu().detach()))
            print('isPos:',bool(label),'true:',bool(true),float(loss))
            if not true:
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
        return key

    def save(self, inputs, preds, targets, _eval):
        pass

class Layer(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size,nStride,relu_param,anti=False):
        super(Layer,self).__init__()
        if anti:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, (kernel_size, kernel_size),stride=nStride, bias=True),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(relu_param),
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, (kernel_size, kernel_size), stride=nStride, bias=True),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(relu_param),
            )
    def forward(self,x):
        return self.net(x)

class generator(nn.Module):
    def __init__(self, kernel_size, input_dim, growth_dim, nStride):
        super(generator, self).__init__()
        self.encoder1_1 = Layer(input_dim, growth_dim, kernel_size, nStride, 0.1, anti=False)
        self.encoder1_2 = Layer(growth_dim, growth_dim*2, kernel_size, nStride, 0.1, anti=False)
        self.encoder1_3 = Layer(growth_dim*2, growth_dim*4, kernel_size, nStride, 0.1, anti=False)

        self.encoder2 = nn.Sequential(
            Layer(input_dim, growth_dim, kernel_size, nStride, 0.1, anti=False),
            Layer(growth_dim, growth_dim*2, kernel_size, nStride, 0.1, anti=False),
            Layer(growth_dim*2, growth_dim*4, kernel_size, nStride, 0.1, anti=False),
        )

        self.decoder_1 = Layer(growth_dim*4, growth_dim*2, kernel_size, nStride, 0.1, anti=True)
        self.decoder_2 = Layer(growth_dim*2, growth_dim, kernel_size, nStride, 0.1, anti=True)
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(growth_dim, input_dim, (kernel_size, kernel_size), stride=nStride, bias=True),
            nn.Tanh(),
        )

    def forward(self, xs):
        x_ori, x_noi = xs
        tmp1 = self.encoder1_1(x_noi)
        tmp2 = self.encoder1_2(tmp1)
        z_noi = self.encoder1_3(tmp2)
        tmp3 = self.decoder_1(z_noi)  + tmp2
        tmp4 = self.decoder_2(tmp3) + tmp1
        x_gen = self.decoder_3(tmp4)
        z_gen = self.encoder2(x_gen)
        return [x_ori, x_gen, z_noi, z_gen]


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
        x_ori, x_gen = xs[0:2]
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
