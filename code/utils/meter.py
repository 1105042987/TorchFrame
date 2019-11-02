class meter(object):
    def __init__(self):
        super(meter,self).__init__()
        self.reset()

    def reset(self):
        self.cnt=0
        self.dic = {}
    
    def add(self,out):
        if self.cnt == 0: 
            self.dic = out
        else: 
            for key,val in out.items():
                self.dic[key] += val
        self.cnt+=1

    def mean(self):
        dic = {}
        if self.cnt == 0:
            raise('No record')
        for key, val in self.dic.items():
            dic[key] = self.dic[key]/self.cnt
        return dic
