import sys
import os
import torch
import pandas as pd

if __name__ == '__main__':
    path = sys.argv[1]
    final = []
    for item in os.listdir(path):
        dic = torch.load(os.path.join(path,item))
        dic['name']=item.replace('.pth','.jpg')
        final.append(dic)
    torch.save(final,os.path.join(path,'sum.pth'))