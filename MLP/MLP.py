import numpy as np

class MLP:
    def __init__(self,input_dim,hidden_dim,out_dim=1,lambda_1=0.0,lambda_2=0.0,dropout=False,prob=0.0):
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.out_dim=out_dim
        self.dropout=dropout
        self.prob=prob

        self.lambda_1=lambda_1
        self.lambda_2=lambda_2
        self.mask=None

        self.params={}
        self.params['w1']=np.random.randn(input_dim,hidden_dim)*np.sqrt(2/input_dim)
        self.params['b1']=np.zeros((1,hidden_dim))
        self.params['w2']=np.random.randn(hidden_dim,out_dim)*np.sqrt(2/hidden_dim)
        self.params['b2']=np.zeros((1,out_dim))


        self.grads={}
    def relu(self,z):
        return np.maximum(0,z)
    
    def sigmoid(self,z):
        clip_z=np.clip(z,-500,500)
        return 1/(1+np.exp(-clip_z))
        
    def forward(self,x,training=True):
        self.X=x
        self.z1=np.dot(self.X,self.params['w1'])+self.params['b1']
        self.a1=self.sigmoid(self.z1)
        self.dropout_active=training and self.dropout and self.prob>0
        if self.dropout_active:
                self.mask=(np.random.rand(*self.a1.shape)>self.prob).astype(float)
    
                self.a1=(self.a1*self.mask)/(1-self.prob)
    
        self.z2=np.dot(self.a1,self.params['w2'])+self.params['b2']
        self.a2=self.sigmoid(self.z2)

      
        return self.a2

        

    def compute_loss(self,y_hat,y):
        N=y.shape[0]
        eps=1e-15
        loss=-(1.0/N)*np.sum(y*np.log(y_hat+eps)+(1.0-y)*np.log(1.0-y_hat+eps))

        l1_panely=0.0
        l2_panelty=0.0

        for key,value in self.params.items():
            if 'b' in key.lower():
                continue
            if self.lambda_1>0:
                l1_panely+=np.sum(np.abs(value))
            if self.lambda_2>0:
                l2_panelty+=np.sum(np.square(value))
        total_loss=loss+(self.lambda_1*l1_panely)+((self.lambda_2/2.0)*l2_panelty)
        return total_loss

    def backward(self,y):
        n=y.shape[0]
        dz2=self.a2-y
        dw2=np.dot(self.a1.T,dz2)/n
        db2=np.sum(dz2,axis=0,keepdims=True)/n

        da1=np.dot(dz2,self.params['w2'].T)
        if self.dropout_active:
            da1=da1*self.mask/(1-self.prob)
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

    def zerograd(self):
        if not self.grads:
            return
        for key in self.grads:
            self.grads[key].fill(0)


def train(model, x_train, y_train, epochs, optimizer, batch_size=None):
    losses = []

    m = x_train.shape[0]

    if batch_size is None:
        batch_size = m

    for epoch in range(epochs):

        indices = np.random.permutation(m)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, m, batch_size):

            x_batch = x_shuffled[i:i + batch_size]
            y_batch = y_shuffled[i:i + batch_size]

            model.zerograd()

            y_pred = model.forward(x_batch, training=True)

            loss = model.compute_loss(y_pred, y_batch)

            model.backward(y_batch)

            optimizer.step()

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")

    return losses  



def test(model, x_test, y_test):
    y_pred = model.forward(x_test, training=False)

    loss = model.compute_loss(y_pred, y_test)

    predictions = (y_pred >= 0.5).astype(int)

    accuracy = np.mean(predictions == y_test)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "predictions": predictions,
        "probabilities": y_pred
    }