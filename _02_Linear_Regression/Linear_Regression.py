# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def main():
    x,y=read_data()
    weight1=ridge(x,y)
    weight2=lasso(x,y)
    
    return 0
     pass
    
def ridge(X_train, X_test, y_train, y_test):
    n_alphas = 200
    alphas = np.logspace(-3,4, n_alphas) #对数等比数列
    clf = linear_model.Ridge(fit_intercept=True)
    coefs = []
    score = []
    intercept = []
    c = 0
    for a in alphas:
        clf.set_params(alpha=a)
        clf.fit(X_train, y_train)
        coefs.append(clf.coef_)
        m = clf.score(X_test,y_test)
        n = clf.intercept_
        score.append(m)
        intercept.append(n)
    for i in range(len(score)):
        if score[i] == max(score):
            c = alphas[i]
            fd = pd.DataFrame({'variable': ['Sex', 'Length', 'Diameter', 'Height', 'Wholeweight', 'Shuckedweight',
                                            'Visceraweight', 'Shellweight'],
                               'weights': coefs[i]})
            print(fd)
            print('intercept:{}\nalphas:{}\nmax(score):{}'.format(intercept[i],alphas[i],score[i]))
    clf1 = linear_model.Ridge(fit_intercept=True)
    clf1.set_params(alpha=c)
    clf1.fit(X_train, y_train)
    y_pre = clf1.predict(X_test)
    # 评价
    # 均方根误差(Root Mean Squared Error, RMSE)
    sum_mean = 0
    for i in range(len(y_pre)):
        sum_mean += (y_pre[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pre))
    print('The value of RMSE in max(score):', sum_erro)
    print('***************************************************')
    # 交叉验证
    clf2 = linear_model.RidgeCV(alphas=alphas)
    clf2.fit(X_train, y_train)
    fr = pd.DataFrame({'variable': ['Sex', 'Length', 'Diameter', 'Height', 'Wholeweight', 'Shuckedweight',
                                    'Visceraweight', 'Shellweight'],
                       'weights': clf2.coef_})
    print(fr)
    print('intercept:{}\nalphas:{}'.format(clf2.intercept_,clf2.alpha_))
    y_pred = clf2.predict(X_test)
    # print(y_pred)  # 1254个样本的预测结果
    # 评价
    # 均方根误差(Root Mean Squared Error, RMSE)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))
    print('the value of RMSE in cross validation:', sum_erro)
    #图形展示
    #设置刻度
    ax = plt.gca()
    #设置刻度的映射
    ax.plot(alphas, score)
    # ax.plot(alphas,coefs)
    #设置x轴的刻度显示方式
    ax.set_xscale('log')
    # #翻转x轴
    # ax.set_xlim(ax.get_xlim()[::-1])
    #设置x、y标签以及标题
    plt.xlabel('alpha')
    plt.ylabel('score')
    # plt.ylabel(('weights'))
    plt.title('Ridge score&alphas')
    # plt.title('Ridge coefficients as a function of the regularization')
    #使得坐标轴最大值和最小值与数据保持一致
    plt.axis('tight')
    plt.show()
    pass
    
def lasso(X_train, X_test, y_train, y_test):
    n_alphas = 200
    alphas = np.logspace(-5, 0, n_alphas)  # 对数等比数列
    coefs = []
    score = []
    intercept = []
    c = 0
    las = Lasso(normalize=True)
    for a in alphas:
        las.set_params(alpha=a)
        las.fit(X_train, y_train)
        coefs.append(las.coef_)
        m = las.score(X_test, y_test)
        n = las.intercept_
        score.append(m)
        intercept.append(n)
    for i in range(len(score)):
        if score[i] == max(score):
            c = alphas[i]
            fd = pd.DataFrame({'variable': ['Sex', 'Length', 'Diameter', 'Height', 'Wholeweight', 'Shuckedweight',
                                            'Visceraweight', 'Shellweight'],
                               'weights': coefs[i]})
            print(fd)
            print('intercept:{}\nalphas:{}\nmax(score):{}'.format(intercept[i], alphas[i], score[i]))
    las1 = Lasso(normalize=True)
    las1.set_params(alpha = c)
    las1.fit(X_train, y_train)
    y_pre = las1.predict(X_test)
    # 评价
    # 均方根误差(Root Mean Squared Error, RMSE)
    sum_mean = 0
    for i in range(len(y_pre)):
        sum_mean += (y_pre[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pre))
    print('The value of RMSE in max(score):', sum_erro)
    print('***************************************************')
    # 交叉验证，找到模型最优的alphas值
    lasso_cv = LassoCV(alphas=alphas, normalize=True, max_iter=1000, cv=None)
    lasso_cv.fit(X_train, y_train)
    fr = pd.DataFrame({'variable': ['Sex', 'Length', 'Diameter', 'Height', 'Wholeweight', 'Shuckedweight',
                                    'Visceraweight', 'Shellweight'],
                       'weights': lasso_cv.coef_})
    print(fr)
    print('intercept:{}\nalphas:{}'.format(lasso_cv.intercept_, lasso_cv.alpha_))
    y_pred = lasso_cv.predict(X_test)
    # print(y_pred)  # 1254个样本的预测结果
    # 评价
    # 均方根误差(Root Mean Squared Error, RMSE)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / 1254)
    print('The value of RMSE in cross validation:', sum_erro)
 
    # 图形展示
    # 设置刻度
    ax = plt.gca()
    # 设置刻度的映射
    ax.plot(alphas, score)
    # ax.plot(alphas,coefs)
    # 设置x轴的刻度显示方式
    ax.set_xscale('log')
    # # 翻转x轴
    # ax.set_xlim(ax.get_xlim()[::-1])
    # 设置x、y标签以及标题
    plt.xlabel('alpha')
    plt.ylabel('score')
    # plt.ylabel(('weights'))
    plt.title('Lasso score&alphas')
    # plt.title('Lasso coefficients as a function of the regularization')
    # 使得坐标轴最大值和最小值与数据保持一致
    plt.axis('tight')
    plt.show()
    return 0
    pass

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y
