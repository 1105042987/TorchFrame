import os
import sys
import json
import json5
import shutil
import argparse
import traceback
import numpy as np
from pprint import pprint
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
    # parser.add_argument('-target_code', action='store_true',
    #                     help='use history code')
    return parser

class Configuration(object):
    def __init__(self,parser, mode):
        super(Configuration,self).__init__()
        self.mode = mode
        args, extras = parser.parse_known_args()
        self.visual = args.visual if 'visual' in args else False
        self.save = args.save if 'save' in args else False

        with open('RootPath.json') as f:
            self.root = json5.load(f)
        if mode == 'train' and args.epoch is None:
            load_path, result_dir = None, None
        else:
            load_path, result_dir = self.__ensure_load_path(args)
        # if args.target_code:
        #     base = sys.path.pop(0)
        #     sys.path.append(os.path.join(result_dir,'code'))

        cfg_path = './config/{}.jsonc'.format(args.cfg_file)
        assert os.path.exists(cfg_path), 'Config file not exist!'
        with open(cfg_path) as f:
            dic = json5.load(f)
        if args.remark is not None:
            dic['system']['remark'] = args.remark
        dic = self.__cfgChange(dic,extras)
        if mode == 'train':
            timestamp = datetime.now().strftime('%m%d_%H%M')
            if len(dic['system']['remark'])>0: timestamp += '_'+dic['system']['remark']
            result_dir = os.path.join(self.root[r'%RESULT%'], args.cfg_file, timestamp)
            self.__copy(args.cfg_file,dic,result_dir)

        self.system = dic['system']
        self.system['result_dir'] = result_dir
        self.system['load_path'] = load_path

        self.optimizer = dic['optimizer']
        self.optimizer['no_opt'] = args.no_opt

        self.dataset = dic['dataset']
        for key,val in dic['dataset'].items():
            if key in ('train','test'): continue
            self.dataset['train'][key] = val
            self.dataset['test'][key] = val

        for key in ('train', 'test'):
            if self.dataset[key]['direction'][0] == '%DATA%':
                self.dataset[key]['direction'][0] = self.root['%DATA%']
            self.dataset[key]['direction'] = os.path.join(*self.dataset[key]['direction'])

    def __cfgChange(self,dic,extras):
        extraParser = argparse.ArgumentParser()
        for arg in extras:
            if arg.startswith(("-", "--")):
                extraParser.add_argument(arg)
        chgDic = vars(extraParser.parse_args(extras))
        for arg,val in chgDic.items():
            for CLS in ('system','optimizer','dataset'):
                if arg in dic[CLS]:
                    try: tmp = int(val)
                    except: 
                        try: tmp = float(val)
                        except: 
                            try: tmp = eval(val)
                            except:
                                tmp = val
                                val = val.lower()
                                if val == 'null' or val == 'none': tmp = None
                                elif val == 'true': tmp = True
                                elif val == 'false': tmp = False
                    if isinstance(tmp,dict):
                        for k,v in tmp.items():
                            dic[CLS][arg][k] = v
                    else:
                        dic[CLS][arg] = tmp

        pprint(dic)
        return dic

    def __copy(self, cfg_name, cfg, result_dir):
        try:
            code_dir = os.path.join(result_dir,'code')
            os.makedirs(os.path.join(result_dir,'ckp'))
            os.makedirs(os.path.join(result_dir,'save'))
            os.makedirs(code_dir)

            tar_config_path = os.path.join(code_dir,'config')
            tar_model_path  = os.path.join(code_dir,'model')
            tar_data_path = os.path.join(code_dir,'dataset')
            os.makedirs(tar_config_path)
            os.makedirs(tar_model_path)
            os.makedirs(tar_data_path)


            with open(os.path.join(tar_config_path,cfg_name+'.jsonc'),'w') as f:
                json.dump(cfg,f)                    # 参数文件保存
            shutil.copy('RootPath.json',code_dir)   # Root 文件保存
            shutil.copy('train.py',code_dir)        # train 文件保存
            shutil.copy('test.py',code_dir)         # test 文件保存
            shutil.copy(os.path.join('model',cfg['system']['net'][0]+'.py'),tar_model_path) # 模型文件保存
            shutil.copy(os.path.join('dataset',cfg['dataset']['file_name']+'.py'),tar_data_path) # 数据集文件保存
            shutil.copytree('docker',os.path.join(code_dir,'docker'))      # docker
        except:
            key = input('\nDo you want to reserve this train(Default False)? y/n: ')
            if key == 'n' and os.path.exists(result_dir): 
                shutil.rmtree(result_dir)
            traceback.print_exc()
            sys.stdout.flush()


    def __ensure_load_path(self, args):
        direct = os.path.join(self.root[r'%RESULT%'], args.cfg_file)
        assert os.path.exists(direct), 'Net {} not exist'.format(args.cfg_file)

        target_timestap_list = os.listdir(direct)
        target_timestap = args.time
        if target_timestap is None:
            target_timestap_list.sort(key=lambda date: date[:9])
            target_timestap = target_timestap_list[-1]
        else:
            for item in target_timestap_list:
                if args.time == item[:len(args.time)]:
                    target_timestap = item
                    break
        direct = os.path.join(direct,target_timestap,'ckp')
        assert os.path.exists(direct), 'Timestamp {} not exist'.format(target_timestap)

        if args.epoch is None:
            epoches = os.listdir(direct)
            if 'best' in epoches:
                args.epoch = 'best'
            else:
                if len(epoches) == 0: raise('Epoch not exist')
                epoches = np.array(epoches, dtype=int)
                args.epoch = str(epoches.max())
        direct = os.path.join(direct, args.epoch)
        assert os.path.exists(direct), 'Epoch {} not exist'.format(args.epoch)

        return direct, os.path.join(self.root[r'%RESULT%'], args.cfg_file, target_timestap)


if __name__ == "__main__":
    pass