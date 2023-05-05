# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def main(data):
    x,y=read_data(path='./data/exp02/')
    print(ridge(x,y,lam=0.2))
    print(lasso(x,y))
    return data @ weight 
     pass
    
def ridge(xArr,yArr,lam=0.2):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    # 矩阵乘法
    xTx = xMat.T * xMat
    rxTx = xTx + lam * np.eye(xMat.shape[1])
    # 判断矩阵是否为可逆矩阵
    if np.linalg.det(rxTx) == 0.0:
        print('This matrix cannot do inverse')
        return

    ws = rxTx.I * xMat.T * yMat
    return ws

    pass
    
def lasso(data):
    return 0
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
