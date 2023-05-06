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
    xTxt = xTx + np.eye(Xarr.shape[1])*len
    if np.linalg.eig(xTxt) == 0.0:
        return
    weight = xTxt.I * Xmat.T * Ymat
  
    return data @ weight
   pass

def lasso(data):
  
    return data @ weight
    pass


def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y


