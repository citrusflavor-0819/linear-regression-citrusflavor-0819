import os
try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    x,y=read_data()
    Xmat = np.mat(x)
    Ymat = np.mat(y)
    xTx = Xmat.T*Xmat
    xTxt = xTx + np.eye(Xarr.shape[1])*(0.2)
    if np.linalg.eig(xTxt) == 0.0:
        return
    weight = xTxt.I * Xmat.T * Ymat
  
    return weight @ data
   pass
   
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    
def lasso(data):

 x,y=read_data()
 Xmat = np.mat(x)
 Ymat = np.mat(y)   
 lr = 0.001
 epochs  =10000  
 m,n = np.shape(x)
 weight= np.mat(np.ones((n,1)))
 
 for i in range(epochs+1):
        h = sigmoid(Xmat*weight)
        weight_grad = Xmat.T*(h - Ymat)/m
        weight = weight - lr *weight_grad
  return weight @ data
    pass


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


