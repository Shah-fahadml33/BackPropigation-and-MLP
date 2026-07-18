
import numpy as np
import math
class SGD:
    def __init__(self,model,lr=0.01):
        self.model=model
        self.lr=lr

    def step(self):
        for key in self.model.perams.keys():
            self.model.perams[key]-=self.lr*self.model.grads[key]
        
        # self.model.perams['w1']-=self.lr*self.model.grades['w1']
        # self.model.perams['b1']-=self.lr*self.model.grades['b1']
        # self.model.perams['w2']-=self.lr*self.model.grades['w2']
        # self.model.permasb2-=self.lr*self.model.grades['b2']


class RMSProp:
    def __init__(self,model,beta=0.9,lr=0.01,eps=1e-8):
        self.lr=lr
        self.beta=beta
        self.model=model
        self.eps=eps
        
        self.v={}
        for key,value in self.model.perams.items():
            self.v[key]=np.zeros_like(value)
        # self.v_w1=np.zeros_like(model.w1)
        # self.v_b1=np.zeros_like(model.b1)
        # self.v_w2=np.zeros_like(model.w2)
        # self.v_b2=np.zeros_like(model.b2)
    def step(self):
        for key in self.model.perams.keys():
            self.v[key]=self.beta*self.v[key]+(1-self.v[key])*(self.model.grads[key]**2)
        # self.v_w1=self.beta*self.v_w1+(1-self.beta)*(self.model.grades['w1']**2)
        # self.v_b1=self.beta*self.v_b1+(1-self.beta)*(self.model.grades['b1']**2)
        # self.v_w2=self.beta*self.v_w2+(1-self.beta)*(self.model.grades['w2']**2)
        # self.v_b2=self.beta*self.v_b2+(1-self.beta)*(self.model.grades['b2']**2)
        self.model.perams[key]-=self.lr*self.model.grads[key]/(np.sqrt(self.v[key])+self.eps)
        # self.model.w1-=self.lr*self.model.grades['w1']/(np.sqrt(self.v_w1)+self.eps)
        # self.model.b2-=self.lr*self.model.grades['b1']/(np.sqrt(self.v_b1)+self.eps)
        # self.model.w2-=self.lr*self.model.grades['w2']/(np.sqrt(self.v_w2)+self.eps)
        # self.model.b2-=self.lr*self.model.grades['b2']/(np.sqrt(self.v_b2)+self.eps)
        

class Adam:
    def __init__(self,model,beta1=0.9,beta2=0.999,lr=0.001,eps=1e-8):
        self.model=model
        self.beta1=beta1
        self.beta2=beta2
        self.lr=lr
        self.eps=eps

        self.t=0

        self.m={}
        self.v={}

        for key,value in self.model.perams.items():
            self.m[key]=np.zeros_like(value)
            self.v[key]=np.zeros_like(value)

    def step(self):
        self.t+=1

        for key in self.model.grads.keys():
            grad=self.model.grads[key]

            self.m[key]=self.beta1*self.m[key]+(1-self.beta1)*grad

            self.v[key]=self.beta2*self.v[key]+((1-self.beta2)*grad**2)

            m_hat=self.m[key]/(1-self.beta1**self.t)
            v_hat=self.v[key]/(1-self.beta2**self.t)

            self.model.perams[key]-=self.lr*m_hat/(np.sqrt(v_hat)+self.eps)


class AdamW:
    def __init__(self,model,beta1=0.9,beta2=0.999,lr=0.001,eps=1e-8,weight_decay=1e-2):
        self.model=model
        self.beta1=beta1
        self.beta2=beta2
        self.lr=lr
        self.eps=eps
        self.weight_decay=weight_decay

        self.m={}
        self.v={}
        self.t=0

        for key,value in self.model.perams.items():
            self.m[key]=np.zeros_like(value)
            self.v[key]=np.zeros_like(value)
    def step(self):
            self.t+=1
            for key in self.model.perams.keys():
                grads=self.model.grads[key]

                self.m[key]=self.beta1*self.m[key]+(1-self.beta1)*grads

                self.v[key]=self.beta2*self.v[key]+((1-self.beta2)*grads**2)

                m_hat=self.m[key]/(1-self.beta1**self.t)
                v_hat=self.v[key]/(1-self.beta2**self.t)

                self.model.perams[key] -= self.lr * (self.weight_decay * self.model.perams[key] + (m_hat / (np.sqrt(v_hat) + self.eps)))
                


