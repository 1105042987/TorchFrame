import json
import argparse
import shutil
import os
from datetime import datetime

def base_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("cfg_file", help="config file", type=str)
    parser.add_argument('-epoch', nargs='?', type=str, default=None,
                        help="Start epoch, Default -1 (train: begin a new training, test: test at the last epoch)")
    parser.add_argument('-time', nargs='?', type=str, default=None,
                        help="timestap you want load from")
    parser.add_argument('-weight_only', action='store_true',
                        help='if you only want to load the weight of net, choose it.')
    parser.add_argument('-result_cfg', action='store_true',
                         help='if you only want to load the cfg_file from the result part, choose it.')
    return parser

class Configuration(object):
    def __init__(self,args,mode):
        super(Configuration,self).__init__()
        try:
            self.mode = mode
            if mode == 'train':
                load_path = None if args.epoch is None else self._ensure_load_path(args,mode)
                timestamp = datetime.now().strftime('%m%d_%H%M')
                result_dir = '../result/{}/{}/'.format(args.cfg_file, timestamp)
                if not os.path.exists(result_dir+'ckp/') and os.path.exists('./config/{}.json'.format(args.cfg_file)):
                    os.makedirs(result_dir+'ckp/')
                    shutil.copy('./config/'+args.cfg_file+'.json', result_dir) # 参数文件保存
                else:
                    raise('Config Does Not Exist!')
            else:
                load_path, result_dir = self._ensure_load_path(args,mode)

            base = result_dir if mode == 'test' and args.result_cfg else './config/'
            with open(base + args.cfg_file + '.json') as f:
                dic = json.load(f)

            self.system = dic['system']
            self.system['result_dir'] = result_dir
            self.system['load_path'] = load_path
            
            self.train = dic['train']
            self.train['weight_only'] = args.weight_only
            
            self.test = dic['test']
            self.dataset = dic['dataset']
            self.dataset['batch_size'] = self.train['batch_size'] if mode == 'train' else self.test['batch_size']
        except:
            if os.path.exists(result_dir): 
                shutil.rmtree(result_dir)
                
        
    def _ensure_load_path(self, args, mode):
        direct = '../result/{}/'.format(args.cfg_file)
        if not os.path.exists(direct):
            print(direct+' not exist')
            raise('Net not exist')

        target_timestap = args.time
        if target_timestap is None:
            target_timestap = os.listdir(direct)
            target_timestap.sort()
            target_timestap = target_timestap[-1]
        direct += '{}/ckp/'.format(target_timestap)
        if not os.path.exists(direct):
            print(direct+' not exist')
            raise('Timestap not exist')

        if args.epoch is None:
            import numpy as np
            epoch = os.listdir(direct)
            if 'best' in epoch:
                args.epoch = 'best'
                epoch.pop(epoch.index('best'))
            else:
                epoch = np.array(epoch, dtype=int)
                if len(epoch) == 0: raise('Epoch not exist')
                args.epoch = epoch.max()
        direct += '{}/'.format(args.epoch)
        if not os.path.exists(direct):
            print(direct+' not exist')
            raise('Epoch not exist')

        if mode == 'test':
            return direct, '../result/{}/{}/'.format(args.cfg_file, target_timestap)
        else:
            return direct
