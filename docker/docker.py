from __future__ import absolute_import
import sys
sys.path.append('..')
import os
import torch
import shutil
import importlib
import traceback
from tqdm import tqdm
from tensorboardX import SummaryWriter
from collections import Iterator
import torch.nn as nn
import gc
from utils.meter import meter

class Docker(object):
    def __init__(self,cfg):
        super(Docker,self).__init__()
        # 模型预设 
        print('Compiling the model ...')
        network_file = 'model.{}'.format(cfg.system['net'])
        dataset_file = 'dataset.{}'.format(cfg.dataset['file_name'])
        network_module = importlib.import_module(network_file)
        dataset_module = importlib.import_module(dataset_file)
        
        self.dev = torch.device('cuda', cfg.system['gpu'][0]) if len(cfg.system['gpu'])>=1 and torch.cuda.is_available() \
            else torch.device('cpu')
        self.multi_dev = len(cfg.system['gpu'])>1

        ## 模型加载
        self.net = network_module.net(**cfg.system['net_param']).to(self.dev)
        self.criterion = network_module.loss(**cfg.system['loss_param']).to(self.dev)
        self.have_loss_weight = cfg.system['have_loss_weight']
        ## 优化器加载
        if cfg.train['optimizer'] == 'adam':
            self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.train['learning_rate'], **cfg.train['adam'])
        elif cfg.train['optimizer'] == 'sgd':
            self.opt = torch.optim.SGD(self.net.parameters(), lr=cfg.train['learning_rate'], **cfg.train['sgd'])
        self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt, cfg.train['milestones'], gamma=cfg.train['decay_rate'], last_epoch=-1)
        self.load_param(cfg)
        ## GPU 分配
        if len(cfg.system['gpu'])>1 and torch.cuda.device_count() > 1:
            self.net = nn.DataParallel(self.net,cfg.system['gpu'])

        # 数据集载入
        print('Loading the dataset ...')
        if cfg.mode == 'train':
            self.end = cfg.train['max_epoch'] + 1
            self.save_epoch = cfg.train['save_epoch']
            self.max_batch = cfg.train['max_batch']
            self.trainloader,self.testloader = dataset_module.dataloader(cfg.dataset,cfg.mode)
            if self.max_batch is None: self.max_batch = len(self.trainloader)
        else:
            self.testloader = dataset_module.dataloader(cfg.dataset,cfg.mode)

        # 评估方式
        self.result_dir = cfg.system['result_dir']
        self.evaluate = network_module.evaluate(result_dir=self.result_dir+'save/',**cfg.system['evaluate_param'])
        # Tensorboard
        self.writer = SummaryWriter(self.result_dir+'tensorboard/')


    def load_param(self,cfg):
        direct = cfg.system['load_path']
        self.best = None
        if direct is None: return

        if cfg.mode == 'test': 
            print('Test at position: '+direct)
        elif not cfg.train['weight_only']:
            other = torch.load(direct+'others.pth', map_location=lambda storage, loc:storage)
            self.opt.load_state_dict(other['opt'])
            self.sch.load_state_dict(other['sch'])
            self.best = other.get('cur_loss',None)

        weight = torch.load(direct+'weight.pth', map_location=lambda storage, loc:storage)
        self.net.load_state_dict(weight)
        if self.have_loss_weight:
            loss = torch.load(direct+'loss.pth', map_location=lambda storage, loc:storage)
            self.criterion.load_state_dict(loss)

    def save(self,epoch,loss_now):
        if epoch == 0:
            temp = self.result_dir+'ckp/best/'
            os.makedirs(temp)
            if self.best is None: self.best = loss_now
            self.__save_param(temp,self.best)
            return 
        if loss_now < self.best:
            self.__save_param(self.result_dir+'ckp/best/', self.best)
        if self.save_epoch == 0: return
        if epoch % self.save_epoch != 0 and epoch != self.end - 1: return 
        temp = self.result_dir+'ckp/{}/'.format(epoch)
        os.makedirs(temp)
        self.__save_param(temp, loss_now)

    def __save_param(self,_dir,_loss):
        if self.multi_dev:
            torch.save(self.net.module.state_dict(), _dir+'weight.pth')
        else:
            torch.save(self.net.state_dict(), _dir+'weight.pth')
        if self.have_loss_weight:
            torch.save(self.criterion.state_dict(), _dir+'loss.pth')
        torch.save({
            'opt': self.opt.state_dict(),
            'sch': self.sch.state_dict(),
            'cur_loss': _loss,
        }, _dir+'others.pth')

    def log_record(self,epoch,idx,dic,board_name=None):
        if board_name is not None:
            log = 'Epoch:{:0>4} '.format(epoch)
            for key,val in dic.items(): 
                log+='{}:{:.5f} '.format(key,val)
            if board_name!='Eval': print(log)
            self.writer.add_scalars(board_name, dic, epoch)
            return 
        log = 'Epoch:{:0>4} Batch:{:0>4} '.format(epoch,idx)
        for key,val in dic.items(): 
            log+='{}:{:.5f} '.format(key,val)
        with open(self.result_dir+'log.txt','a+') as f:
            f.write(log+'\n')
            
    def __step(self,data):
        if self.multi_dev:
            inputs, targets = data[0], to_dev(data[1], self.dev)
        else:
            inputs, targets = to_dev(data,self.dev)
        preds = to_dev(self.net(inputs), self.dev) if self.multi_dev else self.net(inputs)
        inputs[0] = inputs[0].detach().cpu()
        return inputs,preds,targets

    def train(self):
        print('Training begin:')
        try:
            loss_meter = meter()
            main_loss_meter = meter()
            for epoch in range(0, self.end):
                self.net.train()
                sum_dic = {}
                loss_meter.reset()
                main_loss_meter.reset()
                pbar = tqdm(total=self.max_batch, desc='Training Epoch {}'.format(epoch), ascii=True, ncols=130)
                for idx,data in enumerate(self.trainloader):
                    if idx > self.max_batch: break
                    self.opt.zero_grad()
                    if isinstance(data, Iterator):
                        preds, targets = torch.Tensor([]).to(self.dev), torch.Tensor([]).to(self.dev)
                        for d in tqdm(data, leave=False):
                            inputs, sub_pred, sub_tar = self.__step(d)
                            preds = torch.cat([preds, sub_pred],0)
                            targets = torch.cat([targets, sub_tar],0)
                    else:
                        inputs, preds, targets = self.__step(data)
                    loss, record_dic = self.criterion(preds, targets)
                    loss.backward()
                    self.opt.step()

                    loss_meter.add(record_dic)
                    main_loss_meter.add({'main':loss.item()})
                    self.log_record(epoch,idx,record_dic)
                    pbar.set_postfix(record_dic)
                    pbar.update()
                pbar.close()
                self.sch.step()
                if self.testloader is not None:
                    self.log_record(epoch, None, self.test(), 'Eval')
                self.log_record(epoch, None, loss_meter.mean(), 'Loss')
                self.save(epoch,main_loss_meter.mean()['main'])
            self.writer.close()
        except:
            pbar.close()
            traceback.print_exc()
            self.writer.close()
            key = input('\nDo you want to reserve this train(Default False)? y/n: ')
            if key != 'y':
                shutil.rmtree(self.result_dir)


    def test(self,visualize=False,save_result=False):
        print('Testing begin:')
        self.net.eval()
        loss_meter = meter()
        eval_meter = meter()
        pbar = tqdm(total=len(self.testloader), desc='Testing', ascii=True, ncols=130)
        with torch.no_grad():
            for idx, data in enumerate(self.testloader):
                if isinstance(data, Iterator):
                    preds, targets = torch.Tensor([]).to(self.dev), torch.Tensor([]).to(self.dev)
                    for d in tqdm(data, leave=False):
                        inputs, sub_pred, sub_tar = self.__step(d)
                        preds = torch.cat([preds,sub_pred],0)
                        targets = torch.cat([targets,sub_tar],0)
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    inputs,preds,targets = self.__step(data)
                    gc.collect()
                    torch.cuda.empty_cache()
                loss, loss_dic = self.criterion(preds, targets)
                eval_dic = self.evaluate(inputs, preds, targets, visualize, save_result)
                loss_meter.add(loss_dic)
                eval_meter.add(eval_dic)
                pbar.update()
        pbar.close()
        log = 'Test Loss: '
        for key, val in loss_meter.mean().items():
            log += '{}:{:.5f} '.format(key, val)
        log += '\nTest Eval: '
        for key, val in eval_meter.mean().items():
            log += '{}:{:.5f} '.format(key, val)
        print(log)
        return eval_meter.mean()
        


def to_dev(data,dev):
    if type(data) in [tuple,list]:
        return [to_dev(x,dev) for x in data]
    else:
        if type(data) == str:
            return data
        return data.to(dev)


