import numpy as np
def  Bianary_cross_entropy(y_true, y_pred):

    y_pred=np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss=-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1-y_pred))
    return loss

def MSE(y_true, y_pred):
    loss=np.mean((y_true-y_pred)**2)
    return loss

def MAE(y_true, y_pred):
    loss=np.mean(np.abs(y_true-y_pred))
    return loss
def Huber(y_true, y_pred, delta=1.0):
    error=y_true-y_pred
    is_small_error=np.abs(error)<delta
    squared_loss=0.5*error**2
    linear_loss=delta*(np.abs(error)-0.5*delta)
    return np.where(is_small_error,squared_loss,linear_loss)

def cross_entropy(y_true,y_pred):
    y_pred=np.clip(y_pred,1e-15,1-1e-15)
    loss=-np.sum(y_true*np.log(y_pred))
    return loss

def smoothed_cross_entropy(y_pred,smoothed_target):
    batch_size=y_pred.shape[0]    
    y_pred=np.clip(y_pred,1e-15,1-1e-15)
    loss=-np.sum(smoothed_target*np.log(y_pred))/batch_size
    return loss
