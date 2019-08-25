import numpy as np
import math
import keras.backend as K
from keras.callbacks import LambdaCallback
import matplotlib.pyplot as plt

class LRFinder:
    def __init__(self, model):
        self.model = model
        self.lrs = []
        self.losses = []
        self.best_loss = float('inf')
        self.avg_loss = 0.
        
    def on_batch_end(self, batch, logs, beta):
        lr = float(K.get_value(self.model.optimizer.lr))
        loss = logs.get('loss')
        
        # computing moving average of loss
        self.avg_loss = beta * self.avg_loss + (1. - beta) * loss
        smooth_loss = self.avg_loss / (1. - beta**(batch+1)) # batch+1, because the batch number, batch starts from 0
        
        # if loss is NaN or diverges too much, stop training
        if smooth_loss > 4 * self.best_loss or math.isnan(smooth_loss):
            self.model.stop_training=True
            return
        
        # record current stats
        self.lrs.append(lr)
        self.losses.append(smooth_loss)
        
        # update best_loss
        if batch == 1 or smooth_loss < self.best_loss:
            self.best_loss = smooth_loss
            
        # update lr for next batch
        K.set_value(self.model.optimizer.lr, lr*self.lr_multiplier)
        

    def find(self, generator, start_lr = 1e-7, end_lr = 10., beta=0.98, num_iter=100, **kwargs):
        # calculate number of epochs
        num_epochs = math.ceil(num_iter/len(generator))
        
        # calculate lr multiplier
        self.lr_multiplier = (end_lr / start_lr)**(1./num_iter)
        
        # save initial state
        orig_lr = float(K.get_value(self.model.optimizer.lr))
        self.model.save_weights('orig.h5')
        
        # set current lr as start_lr
        K.set_value(self.model.optimizer.lr, start_lr)
        
        # train model with callback based pn self.on_batch_end()
        cb = LambdaCallback(on_batch_end=lambda batch,logs: self.on_batch_end(batch,logs,beta=beta))
        self.model.fit_generator(generator=generator, epochs=num_epochs, callbacks=[cb], **kwargs)
        
        # restore initial state
        K.set_value(self.model.optimizer.lr, orig_lr)
        self.model.load_weights('orig.h5')
            
    def plot_loss(self, skip_start=10, skip_end=5, suggestion=False):
        plt.plot(self.lrs[skip_start:-skip_end], self.losses[skip_start:-skip_end])
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.xscale('log') 
        if suggestion:
            dx = np.log10(self.lr_multiplier) # spacing along log-lr axis
            min_grad_idx = np.argmin(np.gradient(np.array(self.losses[skip_start:-skip_end]), dx))+skip_start
            self.min_loss_grad_lr = self.lrs[min_grad_idx]
            plt.plot(self.lrs[min_grad_idx], self.losses[min_grad_idx], 'ro')
            print("Min. loss grad point, lr: {}, loss: {}".format(self.lrs[min_grad_idx],self.losses[min_grad_idx]))
