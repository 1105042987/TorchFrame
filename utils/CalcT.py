import time
import numpy as np

class CalcTime(object):
    def __init__(self, print_every_toc=True):
        self.__flag = print_every_toc
        self.__num = 0
        self.__nameList = []
        self.__start_dic = {}
        self.__time_dic = {}
    
    def refresh():
        self.__num = 0
        self.__nameList = []
        self.__start_dic = {}
        self.__time_dic = {}

    def tic(self, name=None):
        if name is None:
            name = self.__num
            self.__nameList.append(name)
            self.__num += 1
        else:
            if name not in self.__nameList:
                self.__nameList.append(name)
        self.__start_dic[name] = time.time()

    def toc(self, name=None):
        tmp = time.time()
        if name is None:
            name = self.__nameList.pop()
        else:
            if name in self.__start_dic:
                if name in self.__nameList:
                    self.__nameList.remove(name)
            else:
                raise('Warning: No tic() matched')
        tmp -= self.__start_dic[name]
        if self.__flag:
            print('{} time: {:.4f}s'.format(name,tmp))
        if name in self.__time_dic:
            self.__time_dic[name] = np.append(self.__time_dic[name], tmp)
        else:
            self.__time_dic[name] = np.array([tmp])
        return tmp
            

    def show(self):
        for name in self.__time_dic:
            if len(self.__time_dic[name]) == 1:
                print('{}\t time : {}s'.format(name, np.round(self.__time_dic[name][0],4)))
            else:
                print('{}\t Total: {}s, \t times: {}s'.format(name,
                        np.round(np.sum(self.__time_dic[name]), 4), np.round(self.__time_dic[name], 4),))


if __name__ == "__main__":
    ct = CalcTime()
    ct.tic()
    ct.toc()
