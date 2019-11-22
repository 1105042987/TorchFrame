import json
import argparse
import shutil
import os
import numpy as np
from datetime import datetime

def base_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument('-epoch', nargs='?', type=str, default=None,
                        help="Start epoch, Default -1 (train: begin a new training, test: test at the last epoch)")
    parser.add_argument('-time', nargs='?', type=str, default=None,
                        help="Timestap you want to load from")
    parser.add_argument('-remark', nargs='?', type=str, default=None,
                        help="Remark you want to write")
    parser.add_argument('-no_opt', action='store_true',
                        help='If you do not want to load the weight of optimizer or etc. choose it.')
    parser.add_argument('-target_code', action='store_true',
                        help='If you do not want to load the weight of optimizer or etc. choose it.')
    return parser

class Configuration(object):
    def __init__(self,args, mode):
        super(Configuration,self).__init__()
        self.mode = mode
        with open('RootPath.json') as f:
            self.root = json.load(f)
        if mode == 'train' and args.epoch is None:
            load_path, result_dir = None, None
        else:
            load_path, result_dir = self.__ensure_load_path(args)

        if args.target_code:
            os.chdir(os.path.join(result_dir,'code'))
        cfg_path = './config/{}.json'.format(args.cfg_file)
        assert os.path.exists(cfg_path), 'Config file not exist!'
        with open(cfg_path) as f:
            dic = json.load(f)
        if args.remark is not None:
            dic['system']['remark'] = args.remark

        if mode == 'train':
            timestamp = datetime.now().strftime('%m%d_%H%M')
            if len(dic['system']['remark'])>0: timestamp += '_'+dic['system']['remark']
            result_dir = os.path.join(self.root[r'%RESULT%'], args.cfg_file, timestamp)
            self.__copy(cfg_path,dic,result_dir)

        self.system = dic['system']
        self.system['result_dir'] = result_dir
        self.system['load_path'] = load_path

        self.optimizer = dic['optimizer']
        self.optimizer['no_opt'] = args.no_opt

        self.dataset = dic['dataset']
        for key,val in dic['dataset'].items():
            if key in ['train','test']: continue
            self.dataset['train'][key]=val
            self.dataset['test'][key]=val
        self.dataset['train']['direction'] = self.dataset['train']['direction'].replace('%DATA%',self.root['%DATA%'])
        self.dataset['test']['direction'] = self.dataset['test']['direction'].replace('%DATA%',self.root['%DATA%'])

    def __copy(self, cfg_path, cfg, result_dir):
        try:
            tar_config_path = os.path.join(result_dir,'config')
            tar_model_path  = os.path.join(result_dir,'model')
            tar_data_path = os.path.join(result_dir,'dataset')
            os.makedirs(os.path.join(result_dir,'ckp'))
            os.makedirs(os.path.join(result_dir,'save'))
            os.makedirs(tar_config_path)
            os.makedirs(tar_model_path)
            os.makedirs(tar_data_path)
            shutil.copy('./docker',result_dir)      # docker
            shutil.copy(cfg_path, tar_config_path)  # 参数文件保存
            shutil.copy('RootPath.json',result_dir) # Root 文件保存
            shutil.copy(os.path.join('model',cfg['system']['net'][0]),tar_model_path) # 模型文件保存
            shutil.copy(os.path.join('dataset',cfg['dataset']['file_name']),tar_data_path) # 数据集文件保存
        except:
            if os.path.exists(result_dir): 
                shutil.rmtree(result_dir)



    def __ensure_load_path(self, args):
        direct = os.path.join(self.root[r'%RESULT%'], args.cfg_file)
        assert os.path.exists(direct), 'Net not exist'

        target_timestap_list = os.listdir(direct)
        if target_timestap is None:
            target_timestap_list.sort(key=lambda date: date[:9])
            target_timestap = target_timestap_list[-1]
        else:
            for item in target_timestap_list:
                if args.time == item[:len(args.time)]:
                    target_timestap = item
                    break
        assert os.path.exists(direct), 'Timestamp not exist'

        if args.epoch is None:
            epoches = os.listdir(direct)
            if 'best' in epoches:
                args.epoch = 'best'
            else:
                if len(epoches) == 0: raise('Epoch not exist')
                epoches = np.array(epoches, dtype=int)
                args.epoch = str(epoches.max())
        direct = os.path.join(direct, args.epoch)
        assert os.path.exists(direct), 'Epoch not exist'

        return direct, os.path.join(self.root[r'%RESULT%'], args.cfg_file, target_timestap)
