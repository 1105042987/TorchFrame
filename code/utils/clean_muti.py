import sys
import os
import torch
import shutil


def clean(path):
    if os.path.isdir(path):
        if 'weight.pth' in os.listdir(path):
            print('clean at: ',path)
            w_p = os.path.join(path,'weight.pth')
            weight = torch.load(w_p)
            cleaned = {}
            for key,val in weight.items():
                cleaned[key[7:]] = val
            os.remove(w_p)
            torch.save(cleaned,w_p)
        else:
            for item in os.listdir(path):
                clean(os.path.join(path,item))



if __name__ == '__main__':
    path = sys.argv[1]
    clean(path)
    