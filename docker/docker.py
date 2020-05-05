from __future__ import absolute_import
import sys
base = sys.path[0]
sys.path.append(os.path.abspath(os.path.join(base, "..")))
import os
import torch
import shutil
import importlib
import traceback
from tqdm import tqdm
from os.path import join as PJOIN
from tensorboardX import SummaryWriter
from collections import Iterator
from docker.meter import meter
import torch.nn as nn
import gc

class Docker(object):
    def __init__(self,cfg):
        super(Docker,self).__init__()
        # 模型预设 
        print('Compiling the model ...')
        network_file = 'model.{}'.format(cfg.system['net'][0])
        dataset_file = 'dataset.{}'.format(cfg.dataset['file_name'])
        network_module = importlib.import_module(network_file)
        dataset_module = importlib.import_module(dataset_file)
        
        self.dev = torch.device('cuda', cfg.system['gpu'][0]) if len(cfg.system['gpu'])>=1 and torch.cuda.is_available() \
            else torch.device('cpu')
        self.multi_dev = len(cfg.system['gpu'])>1
        self.epoch = 'test'

        ## 模型加载
        self.net = getattr(network_module,cfg.system['net'][1])(**cfg.system['net_param']).to(self.dev)
        self.criterion = network_module.loss(**cfg.system['loss_param'])
        ## 优化器加载
        if cfg.mode == 'train':
            self.best = None
            self.epoch_start = 1
            self.eval_on_train = cfg.optimizer['eval_on_train']
            self.epoch_end = cfg.optimizer['max_epoch'] + 1
            self.save_epoch = cfg.optimizer['save_epoch']
            self.max_batch = cfg.optimizer['max_batch']
            if cfg.optimizer['type'] == 'adam':
                self.opt = torch.optim.Adam(self.net.parameters(), 
                                lr=cfg.optimizer['learning_rate'], **cfg.optimizer['adam'])
            elif cfg.optimizer['type'] == 'sgd':
                self.opt = torch.optim.SGD(self.net.parameters(), 
                                lr=cfg.optimizer['learning_rate'], **cfg.optimizer['sgd'])
            self.sch = torch.optim.lr_scheduler.MultiStepLR(self.opt, cfg.optimizer['milestones'],
                                gamma=cfg.optimizer['decay_rate'], last_epoch=-1)
        self.load_param(cfg)
        ## GPU 分配
        if self.multi_dev and torch.cuda.device_count()>1:
            self.net = nn.DataParallel(self.net,cfg.system['gpu'])

        # 数据集载入
        print('Loading the dataset ...')
        if cfg.mode == 'train':
            self.trainloader = dataset_module.dataloader(cfg.dataset[cfg.mode],cfg.mode)
            if self.max_batch is None: self.max_batch = len(self.trainloader)
            self.testloader = dataset_module.dataloader(cfg.dataset['test'],
                                'test') if cfg.optimizer['test_on_train'] else None
        else:
            self.testloader = dataset_module.dataloader(cfg.dataset[cfg.mode],cfg.mode)

        # 评估方式
        self.result_dir = cfg.system['result_dir']
        self.evaluate = network_module.evaluate(**cfg.system['evaluate_param'])
        self.evaluate.result_dir = PJOIN(self.result_dir,'save')
        # Tensorboard
        self.writer = SummaryWriter(PJOIN(self.result_dir,'tensorboard')) if cfg.mode == 'train' else None

    def load_param(self,cfg):
        direct = cfg.system['load_path']
        if direct is None: return
        if cfg.mode == 'test': 
            print('Test at position: '+direct)
        elif not cfg.optimizer['no_opt']:
            other = torch.load(PJOIN(direct,'others.pth'), map_location=lambda storage, loc:storage)
            self.opt.load_state_dict(other['opt'])
            self.sch.load_state_dict(other['sch'])
            self.best = other.get('cur_loss', None)
            self.epoch_start += other.get('epoch',0)
            self.epoch_end += other.get('epoch',0)
        weight = torch.load(PJOIN(direct,'weight.pth'), map_location=lambda storage, loc:storage)
        self.net.load_state_dict(weight)

    def save(self,loss_now):
        best_save = PJOIN(self.result_dir,'ckp','best')
        if self.epoch == self.epoch_start:
            os.makedirs(best_save)
            if self.best is None: self.best = loss_now
            self.__save_param(best_save,self.best)
            return
        if loss_now < self.best:
            self.__save_param(best_save, self.best)
        if self.epoch != self.epoch_end-1:
            if self.save_epoch == 0: return                  # Just save the best
            if self.epoch % self.save_epoch != 0: return     # Save every save epoch
        now_save = PJOIN(self.result_dir,'ckp',str(self.epoch))
        os.makedirs(now_save)
        self.__save_param(now_save, loss_now)

    def log_record(self,dic,board_name):
        log = 'Epoch:{:0>4} '.format(self.epoch)
        for key,val in dic.items():
            log+='{}:{:.5f} '.format(key,val)
        print(board_name,log)
        if self.writer is not None:
            with open(PJOIN(self.result_dir,board_name+'_log.txt'),'a+') as f:
                f.write(log+'\n')
            self.writer.add_scalars(board_name, dic, self.epoch) 
            
    def train(self):
        print('Training begin:')
        try:
            loss_meter = meter()
            main_loss_meter = meter()
            eval_meter = meter()
            for self.epoch in range(self.epoch_start, self.epoch_end):
                self.net.train()
                loss_meter.reset()
                main_loss_meter.reset()
                eval_meter.reset()
                pbar = tqdm(total=self.max_batch, desc='Training Epoch {}'.format(self.epoch), ascii=True, ncols=130)
                for idx,data in enumerate(self.trainloader):
                    if idx >= self.max_batch: break
                    self.opt.zero_grad()
                    if isinstance(data, Iterator):
                        preds, targets = [], []
                        for d in tqdm(data, ascii=True, leave=False, ncols=130):
                            inputs, sub_pred, sub_tar = self.__step(d)
                            preds.append(sub_pred)
                            targets.append(sub_tar)
                        preds = torch.cat(preds,0)
                        targets = torch.cat(targets,0)
                    else:
                        inputs, preds, targets = self.__step(data)
                    loss, record_dic = self.criterion(preds, targets)
                    loss.backward()
                    self.opt.step()

                    if self.eval_on_train:
                        eval_dic = self.evaluate(inputs, preds, targets, False, False)
                        eval_meter.add(eval_dic)
                    loss_meter.add(record_dic)
                    main_loss_meter.add({'main':loss.item()})
                    pbar.set_postfix(record_dic)
                    pbar.update()
                pbar.close()
                self.sch.step()
                self.log_record(loss_meter.mean(), 'Train_Loss')
                if self.eval_on_train:
                    self.log_record(eval_meter.mean(), 'Train_Eval')
                if self.testloader is not None:
                    self.test(False,False)
                self.save(main_loss_meter.mean()['main'])
            self.writer.close()
        except:
            pbar.close()
            self.writer.close()
            traceback.print_exc()
            sys.stdout.flush()
            key = input('\nDo you want to reserve this train(Default False)? y/n: ')
            if key != 'y':
                shutil.rmtree(self.result_dir)


    def test(self,visualize=False,save_result=False):
        self.net.eval()
        loss_meter = meter()
        eval_meter = meter()
        pbar = tqdm(total=len(self.testloader), desc='Testing', ascii=True, ncols=130)
        with torch.no_grad():
            for idx, data in enumerate(self.testloader):
                if isinstance(data, Iterator):
                        preds, targets = [], []
                        for d in tqdm(data, ascii=True, leave=False, ncols=130):
                            inputs, sub_pred, sub_tar = self.__step(d)
                            preds.append(sub_pred)
                            targets.append(sub_tar)
                            gc.collect()
                            torch.cuda.empty_cache()
                        preds = torch.cat(preds,0)
                        targets = torch.cat(targets,0)
                else:
                    inputs,preds,targets = self.__step(data)
                    gc.collect()
                    torch.cuda.empty_cache()
                loss, loss_dic = self.criterion(preds, targets)
                eval_dic = self.evaluate(inputs, preds, targets, visualize, save_result)
                loss_meter.add(loss_dic)
                eval_meter.add(eval_dic)
                pbar.set_postfix(loss_dic)
                pbar.update()
        pbar.close()
        self.log_record(loss_meter.mean(),'Test_Loss')
        self.log_record(eval_meter.mean(),'Test_Eval')
        return eval_meter.mean()

    def __save_param(self,_dir,_loss):
        if self.multi_dev:
            torch.save(self.net.module.state_dict(), PJOIN(_dir,'weight.pth'))
        else:
            torch.save(self.net.state_dict(), PJOIN(_dir,'weight.pth'))
        torch.save({
            'opt': self.opt.state_dict(),
            'sch': self.sch.state_dict(),
            'epoch':self.epoch,
            'cur_loss': _loss,
        }, PJOIN(_dir,'others.pth'))

    def __step(self,data):
        if self.multi_dev:
            inputs, targets = data[0], to_dev(data[1], self.dev)
        else:
            inputs, targets = to_dev(data,self.dev)
        preds = to_dev(self.net(inputs), self.dev) if self.multi_dev else self.net(inputs)
        return inputs,preds,targets
        


def to_dev(data,dev):
    if type(data) in [tuple,list]:
        return [to_dev(x,dev) for x in data]
    else:
        if type(data) == str:
            return data
        return data.to(dev)


