import keras.backend as K
from keras.callbacks import Callback
from param_scheduler import CosineScheduler
import matplotlib.pyplot as plt

class OneCycleScheduler(Callback):
    
    def __init__(self, max_lr, momentums=(0.95,0.85), start_div=25., pct_start=0.3, verbose=True, sched=CosineScheduler, end_div=None):
        self.max_lr, self.momentums, self.start_div, self.pct_start, self.verbose, self.sched, self.end_div = max_lr, momentums, start_div, pct_start, verbose, sched, end_div
        if self.end_div is None:
            self.end_div = start_div * 1e4
        self.logs = {}
        
    def on_train_begin(self, logs=None):
        self.num_epochs = self.params['epochs']
        self.steps_per_epoch = self.params['steps']
        start_lr = self.max_lr/self.start_div
        end_lr = self.max_lr/self.end_div
        num_iter = self.num_epochs * self.steps_per_epoch
        num_iter_1 = int(self.pct_start*num_iter)
        num_iter_2 = num_iter - num_iter_1
        self.lr_scheds = (self.sched(start_lr, self.max_lr, num_iter_1), self.sched(self.max_lr, end_lr, num_iter_2))
        self.momentum_scheds = (self.sched(self.momentums[0], self.momentums[1], num_iter_1), self.sched(self.momentums[1], self.momentums[0], num_iter_2))
        self.sched_idx = 0
        self.optimizer_params_step()   
        
    def optimizer_params_step(self):
        next_lr = self.lr_scheds[self.sched_idx].step()
        next_momentum = self.momentum_scheds[self.sched_idx].step()
        
        # add to logs
        self.logs.setdefault('lr', []).append(next_lr)
        self.logs.setdefault('momentum', []).append(next_momentum)
        
        # update optimizer params
        K.set_value(self.model.optimizer.lr, next_lr)
        if hasattr(self.model.optimizer, 'momentum'):
            K.set_value(self.model.optimizer.momentum, next_momentum)
        
    def on_batch_end(self, batch, logs=None):
        if self.sched_idx >= len(self.lr_scheds):
            self.model.stop_training=True
            return
        self.optimizer_params_step()
        if self.lr_scheds[self.sched_idx].is_complete():
            self.sched_idx += 1
            
    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if hasattr(self.model.optimizer, 'momentum'):
                print("- OneCycleScheduler, lr: {}, momentum: {}".format(self.logs['lr'][-1], self.logs['momentum'][-1]))
            else:
                print("- OneCycleScheduler, lr: {}".format(self.logs['lr'][-1]))
            
        if epoch >= self.num_epochs:
            self.model.stop_training=True
            return
        
    def plot_lr(self, show_momentums=True):
        if hasattr(self.model.optimizer, 'momentum') and show_momentums:
            plt.subplot(121)
            plt.plot(self.logs['lr'])
            plt.ylabel('learning rate')
            plt.xlabel('iteration') 

            plt.subplot(122)
            plt.plot(self.logs['momentum'])
            plt.ylabel('momentum')
            plt.xlabel('iteration') 
        else:
            plt.plot(self.logs['lr'])
            plt.ylabel('learning rate')
            plt.xlabel('iteration')
        
