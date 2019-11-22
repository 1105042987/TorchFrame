import time
import numpy as np

class CalcTime(object):
    def __init__(self, print_every_toc=True, between_toc=False):
        self.__print_every_toc = print_every_toc
        self.__between_toc = between_toc
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
        last = 0
        if name in self.__time_dic:
            last = self.__time_dic[name][-1]
            self.__time_dic[name] = np.append(self.__time_dic[name], tmp)
        else:
            self.__time_dic[name] = np.array([tmp])
        if self.__print_every_toc:
            if self.__between_toc:
                print('{} time: {:.4f}s'.format(name,tmp-last))
            else:
                print('{} time: {:.4f}s'.format(name,tmp))
        return tmp
            

    def show(self,delta_time=False):
        np.set_printoptions(threshold=np.nan)
        for name in self.__time_dic:
            if len(self.__time_dic[name]) == 1:
                print('{}\t time : {}s'.format(name, np.round(self.__time_dic[name][0],4)))
            else:
                if delta_time:
                    delta_time = self.__time_dic[name].copy()
                    delta_time[1:] -= self.__time_dic[name][:-1]
                    print('{}\t Total: {}s, Mean: {}s \t delta_times: {}s'.format(name,
                            np.round(np.sum(delta_time), 4), np.round(np.mean(delta_time), 4), np.round(delta_time, 4),))
                else:
                    print('{}\t cumul_times: {}s'.format(name, np.round(self.__time_dic[name], 4),))
                    


if __name__ == "__main__":
    ct = CalcTime(False)
    ct.tic()
    ct.toc(0)
    time.sleep(0.1)
    ct.toc(0)
    time.sleep(0.2)
    ct.toc(0)
    ct.show()
    ct.show(True)
