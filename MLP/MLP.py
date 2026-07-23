import numpy as np

class MLP:
    def __init__(self,input_dim,hidden_dim,out_dim=1,lambda_1=0.0,lambda_2=0.0):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim

        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        self.params={}
        self.params['w1']=np.random.randn(input_dim,hidden_dim)*np.sqrt(2/input_dim)
        self.params['b1']=np.zeros((1,hidden_dim))
        self.params['w2']=np.random.randn(hidden_dim,out_dim)*np.sqrt(2/hidden_dim)
        self.params['b2']=np.zeros((1,out_dim))


        self.grads={}
    def relu(self,z):
        return np.max(0,z)
    
    def sigmoid(self,z):
        clip_z=np.clip(z,-500,500)
        return 1/(1+np.exp(-clip_z))
        
    def forward(self,x):
        self.X=x
        self.z1=np.dot(self.X,self.params['w1'])+self.params['b1']
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.params['w2'])+self.params['b2']
        self.a2=self.sigmoid(self.z2)

        return self.a2

    def compute_loss(self,y_hat,y):
        N=y.shape[0]
        eps=1e-15
        loss=-(1.0/N)*np.sum(y*np.log(y_hat+eps)+(1.0-y)*np.log(1.0-y_hat+eps))

        l1_panely=0.0
        l2_panelty=0.0

        for key,value in self.params.item():
            if 'b' in key.lower():
                continue
            if self.lambda_1>0:
                l1_panely+=np.sum(np.abs(value))
            if self.lambda_2>0:
                l2_panelty+=np.sum(np.square(value))
        total_loss=loss+(self.lambda_1*l1_panely)+((self.lambda_2/2.0)*l2_panelty)
        return loss

    def backward(self,y):
        n=y.shape[0]
        dz2=self.a2-y
        dw2=np.dot(self.a1.T,dz2)/n
        db2=np.sum(dz2,axis=0,keepdims=True)/n

        da1=np.dot(dz2,self.params['w2'].T)
        dz1=da1* self.a1 *(1.0-self.a1)
        dw1=np.dot(self.X.T,dz1)/n
        db1=np.sum(dz1,axis=0,keepdims=True)/n
        if self.lambda_1>0:
            dw1+=self.lambda_1*np.sign(self.params['w1'])
            dw2+=self.lambda_1*np.sign(self.params['w2'])
        if self.lambda_2>0:
            dw1+=self.lambda_2*self.params['w1']
            dw2+=self.lambda_2*self.params['w2']
        self.grads['w1']=dw1
        self.grads['b1']=db1
        self.grads['w2']=dw2
        self.grads['b2']=db2
        return self.grads

    def get_params_vector(self):
        return np.concatenate([
            self.params['w1'].ravel(),self.params['b1'].ravel(),
            self.params['w2'].ravel(),self.params['b2'].ravel()
        ])

    def set_params_from_vector(self,theta):
        w1_end=self.input_dim*self.hidden_dim
        b1_end=w1_end+self.hidden_dim
        w2_end=b1_end+(self.hidden_dim*self.out_dim)

        self.params['w1']=theta[:w1_end].reshape(self.input_dim,self.hidden_dim)
        self.params['b1']=theta[w1_end:b1_end].reshape((1,self.hidden_dim))
        self.params['w2']=theta[b1_end:w2_end].reshape(self.hidden_dim,self.out_dim)
        self.params['b2']=theta[w2_end:].reshape((1,self.out_dim))

    def set_grads_vecter(self):
        return np.concatenate([
            self.grads['w1'].ravel(),self.grads['b1'].ravel(),
            self.grads['w2'].ravel(),self.grads['b2'].ravel()
        ])
    

def train(model,x_train,y_train,epochs,batch_size=None,optimizer=None):
    losses=[]
    if batch_size is None:
        for epoch in range(epochs):
            preds=model.forward(x_train)
            loss=model.compute_loss(preds,y_train)
            losses.append(loss)
            model.backward(y_train)
            optimizer.step()

            if epoch %100 ==0:
                print(f"Epoch {epoch}  | Loss :  {loss}")
        return losses
    if batch_size:
        m,n=x_train.shape
        
        for epoch in range(epochs):
            num_batches=0
            epoch_loss=0.0
            indices=np.random.permutation(m)
            x_shuffled=x_train[indices]
            y_shuffled=y_train[indices]
             
            for i in range(0,m,batch_size):
                x_batch=x_shuffled[i:i+batch_size]
                y_batch=y_shuffled[i:i+batch_size]

                pred=model.forward(x_batch)
                loss=model.compute_loss(pred,y_batch)
                epoch_loss+=loss
                num_batches+=1
                
                model.backward()
                optimizer.step()
            losses.append(epoch_loss/num_batches)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss: {epoch_loss/num_batches:.4f}")
    return losses

    
def test(model,x_test,y_test,batch_size=None):
    losses=[]
    if batch_size is None:
        preds=model.forward(x_test)
        test_loss=model.compute_loss(preds,y_test)
        losses.append(test_loss)

        print(f"Test Loss is :{test_loss}")
        return preds,losses
    else:
        m,n=x_test.shape
        epoch_los=0.0
        num_batches=0.0
        prediction=[]
        for i  in range(0,m,batch_size):
            x_batch=x_test[i:i+batch_size]
            y_batch=y_test[i:i+batch_size]

            preds=model.forward(x_batch)
            prediction.append(preds)
            loss=model.compute_loss(preds,y_batch)

            epoch_los+=loss
            num_batches+=1
        avg_loss=epoch_los/num_batches
        losses.append(avg_loss)
        print(f"Test Loss is :{avg_loss}")
        return preds,losses

