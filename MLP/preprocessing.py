import numpy as np
def smooth_label(labels,num_classes,alpha=0.1):
    """ we are applying here the mathematical formula for label smoothing, which is given by:
    smoothed_label = (1 - alpha) * one_hot_label + alpha / num_classes
    
    """
    batch_size=labels.shape[0]
    smooth_val=alpha/num_classes
    smothed_target=np.full((batch_size,num_classes),smooth_val)
    true_class_val=(1-alpha)+smooth_val
    smothed_target[np.arange(batch_size),labels]=true_class_val

    return smothed_target
