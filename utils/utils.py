import torch
def calcAUC(pre,tar):
    assert pre.shape==tar.shape
    B = pre.shape[0]
    auc = torch.zeros(B)
    for i in range(B):
        auc_list = tar[i].reshape(-1)[pre[i].reshape(-1).sort(descending=True)[1]]
        Tru,Fal = int(tar[i].sum()),int((1-tar[i]).sum())
        rank = auc_list.sort(descending=True)[1]
        auc[i]=float(rank[:Tru].sum()-Tru*(Tru+1)/2)/float(Tru*Fal)
    return auc.mean()

if __name__ == "__main__":
    a=torch.rand(3,8)
    b=torch.randint(2,(3,8)).float()
    calcAUC(a,b)