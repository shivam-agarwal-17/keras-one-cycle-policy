import numpy as np

class ParamScheduler:
    def __init__(self, start, end, num_iter):
        self.start = start
        self.end = end
        self.num_iter = num_iter
        self.idx = -1
        
    def func(self, start_val, end_val, pct):
        raise NotImplementedError
        
    def step(self):
        self.idx+=1
        return self.func(self.start, self.end, self.idx/self.num_iter)
    
    def reset(self):
        self.idx=-1
        
    def is_complete(self):
        return self.idx >= self.num_iter

class LinearScheduler(ParamScheduler):
    
    def func(self, start_val, end_val, pct):
        return start_val + pct * (end_val - start_val)
    
class CosineScheduler(ParamScheduler):
    
    def func(self, start_val, end_val, pct):
        cos_out = np.cos(np.pi * pct) + 1
        return end_val + (start_val - end_val)/2 * cos_out
