# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def main():
    x,y=read_data()
    print(ridge(x，y))
    print(lasso(x,y))
     pass
    
def ridge(x,y,len=0.2):
    Xmat = np.mat(x)
    Ymat = np.mat(y)
    xTx = Xmat.T*Xmat
    xTxt = xTx + np.eye(x.shape[1])*len
    if np.linalg.eig(xTxt) == 0.0:
        return
    ws = xTxt.I * Xmat.T * Ymat
    return ws
    pass
    
def lasso(data):
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
