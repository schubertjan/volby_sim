import numpy as np

def logit(x):
    #make sure it works even if not a np array supplied
    x_ = [1e-4 if i==0 else i for i in x]    
    #make an array
    x_ = np.array(x_)
    l = np.log(x_ / (1-x_))
    return l

def sigmoid(x):
    s = 1 / (1 + np.exp(1) ** -x)
    return s
    