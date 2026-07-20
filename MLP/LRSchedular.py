import numpy as np
class WarmupCosineSchedular:
    def __init__(self,peak_lr,warmup_step,total_steps,min_lr=0):
        self.peak_lr=peak_lr
        self.warmup_step=warmup_step
        self.total_steps=total_steps
        self.min_lr=min_lr
        self.current_step=0

    def get_lr(self):
        if self.current_step<=self.warmup_steps:
            if self.warmup_steps==0:
                return self.peak_lr
            return self.peak_lr*(self.current_lr/self.warmup_steps)
        decay_steps=self.total_steps-self.warmup_steps
        curr_decay_step=self.curr_step-self.warmup_steps
        decay_ratio=curr_decay_step/decay_steps

        coef=0.5*(1.0+np.cos(np.pi*decay_ratio))

        return self.min_lr+coef*(self.peak_lr-self.min_lr)
    
    def step(self):
        lr=self.get_lr()
        self.current_step+=1
        return lr
    

class StepDecaySchedular:
    def __init__(self,initial_lr,step_size,gamma=0.5):
        self.initial_lr=initial_lr
        self.step_size=step_size
        self.gamma=gamma
        self.epoch=0

    def step(self):
        self.epoch+=1
        power=np.floor(self.epoch/self.step_size)
        self.current_lr=self.initial_lr*(self.gamma**power)

        return self.current_lr
def get_lr(self):
    return self.current_lr


class ReduceLRonPlateau:
    def __init__(self,initial_lr,factor=0.1,patience=10,min_lr=0.1):
        self.current_lr=self.current_lr
        self.factor=factor
        self.patience=patience
        self.min_lr=min_lr
        self.best_loss=np.inf
        self.wait_count=0
        
    def step(self,current_loos):
        if current_loos<self.best_loss:
            self.best_loss=current_loos
            self.wait_count=0
        else:
            self.wait_count+=1
        
        if self.wait_count>=self.patience:
            new_lr=self.current_lr*self.factor
            self.current_lr=max(new_lr,self.min_lr)

        return self.current_lr
    def get_lr(self):
        return self.current_lr
