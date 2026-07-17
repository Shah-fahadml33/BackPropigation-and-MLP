import numpy as np

class MLP:
    def __init__(self,input_dim,hidden_dim,out_dim=1):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim

        self.perams={}
        self.perams['w1']=np.random.randn(input_dim,hidden_dim)*np.sqrt(2/input_dim)
        self.perams['b1']=np.zeros((1,hidden_dim))
        self.perams['w2']=np.random.randn(hidden_dim,out_dim)*np.sqrt(2/hidden_dim)
        self.perams['b2']=np.zeros((1,out_dim))


        self.grads={}
    def sigmoid(self,z):
        clip_z=np.clip(z,-500,500)
        return 1/(1+np.exp(-clip_z))
        
    def forward(self,x):
        self.X=x
        self.z1=np.dot(self.X,self.perams['w1'])+self.perams['b1']
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.perams['w2'])+self.perams['b2']
        self.a2=self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self,y_hat,y):
        N=y.shape[0]
        eps=1e-15
        loss=-(1.0/N)*np.sum(y*np.log(y_hat+eps)+(1.0-y)*np.log(1.0-y_hat+eps))
        return loss

    def backward(self,y):
        n=y.shape[0]
        dz2=self.a2-y
        dw2=np.dot(self.a1.T,dz2)/n
        db2=np.sum(dz2,axis=0,keepdims=True)/n

        da1=np.dot(dz2,self.perams['w2'].T)
        dz1=da1* self.a1 *(1.0-self.a1)
        dw1=np.dot(self.X.T,dz1)/n
        db1=np.sum(dz1,axis=0,keepdims=True)/n

        self.grads['w1']=dw1
        self.grads['b1']=db1
        self.grads['w2']=dw2
        self.grads['b2']=db2
        return self.grads

    def get_perams_vector(self):
        return np.concatenate([
            self.perams['w1'].ravel(),self.perams['b1'].ravel(),
            self.perams['w2'].ravel(),self.perams['b2'].ravel()
        ])

    def set_perams_from_vector(self,theta):
        w1_end=self.input_dim*self.hidden_dim
        b1_end=w1_end+self.hidden_dim
        w2_end=b1_end+(self.hidden_dim*self.out_dim)

        self.perams['w1']=theta[:w1_end].reshape(self.input_dim,self.hidden_dim)
        self.perams['b1']=theta[w1_end:b1_end].reshape((1,self.hidden_dim))
        self.perams['w2']=theta[b1_end:w2_end].reshape(self.hidden_dim,self.out_dim)
        self.perams['b2']=theta[w2_end:].reshape((1,self.out_dim))

    def set_grads_vecter(self):
        return np.concatenate([
            self.grads['w1'].ravel(),self.grads['b1'].ravel(),
            self.grads['w2'].ravel(),self.grads['b2'].ravel()
        ])
    