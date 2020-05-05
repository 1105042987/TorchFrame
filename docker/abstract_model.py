from abc import ABCMeta, abstractmethod
import torch.nn as nn
import torch,numpy 
import os

class weak_loss(nn.Module,metaclass=ABCMeta):
    def __init__(self):
        super(weak_loss, self).__init__()
    
    @abstractmethod
    def get_loss(self,preds,targets):
        # return "a loss for backward" and "a dic for recording loss"
        return None,{}

    def forward(self,preds,targets):
        loss, dic = self.get_loss(preds, targets)
        for key,val in dic.items():
            dic[key] = to_float(val)
        return loss,dic

class weak_evaluate(metaclass=ABCMeta):
    def __init__(self):
        super(weak_evaluate, self).__init__()
        self.continue_visual = True
        self.first_enter = True
        self.result_dir = None

    def __call__(self,inputs,preds,targets,visual=False,save=False):
        _eval = self.get_eval(inputs,preds,targets)
        for key,val in _eval.items():
            _eval[key] = to_float(val)
        if save: self.save(inputs, preds, targets, _eval)
        if not visual: return _eval

        if self.first_enter:
            key = input("\nif you want quit, input 'q'.\n# Any key to continue#")
            self.first_enter = False
            if key == 'q': self.continue_visual = False
        log = ''
        if self.continue_visual:
            key = self.visualize(inputs,preds,targets,_eval)
            if key is None: 
                for key, val in _eval.items(): log += '{}:{:.5f} '.format(key, val)
                key = input(log)
            if key == ord('q') or key == 'q': 
                self.continue_visual = False
        return _eval

    @abstractmethod
    def get_eval(self,inputs,preds,targets):
        return {}

    @abstractmethod
    def visualize(self,inputs,preds,targets,_eval):
        return None

    @abstractmethod
    def save(self,inputs,preds,targets,_eval):
        pass


class weak_SplitPatch(metaclass=ABCMeta):
    def __init__(self,w,h,patch_size,stride,batch):
        self.WS, self.WE = self.StartEndSplit(w, patch_size, stride)
        self.HS, self.HE = self.StartEndSplit(h, patch_size, stride)
        self.batch = batch
        self.len = len(self.WS)*len(self.HS)
        self.len = self.len//self.batch+((self.len % self.batch) != 0)
        self.num_cnt = 0

    @staticmethod
    def StartEndSplit(whole_len, patch_size, stride):
        if whole_len < patch_size:
            whole_len = patch_size
        start = list(range(0, whole_len - patch_size + 1, stride))
        end = list(range(patch_size, whole_len + 1, stride))
        if end[-1] < whole_len:
            end.append(whole_len)
            start.append(whole_len-patch_size)
        return start,end

    def __iter__(self):
        self.GEN = self.Generator()
        return self

    def __next__(self):
        inps,tars = [],[]
        cnt=0
        if self.num_cnt < self.len:
            self.num_cnt += 1
            for inp,tar in self.GEN:
                inps.append(inp)
                tars.append(tar)
                cnt+=1
                if cnt==self.batch: break
            return torch.cat(inps),torch.cat(tars)
        else:
            raise StopIteration

    def Generator(self):
        for ws, we in zip(self.WS,self.WE):
            for hs, he in zip(self.HS,self.HE):
                inp = self.get_input(ws,we,hs,he).unsqueeze(0)
                tar = self.get_target(ws,we,hs,he).unsqueeze(0)
                yield inp,tar

    @abstractmethod
    def get_input(self, we, ws, hs, he):
        return torch.Tensor([])

    @abstractmethod
    def get_target(self, we, ws, hs, he):
        return torch.Tensor([])

    def __len__(self):
        return self.len


def to_float(data):
    try:
        return float(data)
    except:
        return [to_float(x) for x in data]

if __name__ == "__main__":
    class net(nn.Module):
        def __init__(self):
            super(net,self).__init__()
        def forward(self,xs):
            pass
    class loss(weak_loss):
        def __init__(self):
            super(loss,self).__init__()
        def get_loss(self,pre,tar):
            return None,{}
    class evaluate(weak_evaluate):
        def __init__(self,result_dir):
            super(evaluate,self).__init__(result_dir)
        def get_eval(self, inputs, preds, targets):
            return {}
        def visualize(self, inputs, preds, targets, _eval):
            return None
        def save(self, inputs, preds, targets, _eval):
            pass

    a=test()
    print(a(1,2))


