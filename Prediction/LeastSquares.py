#coding=UTF-8
'''
Created on 2015年12月27日

@author: 15361
最小二乘法
'''
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    X=[]
    Y=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        X.append(float(curline[0]))
        Y.append(float(curline[1]))
    return X,Y

def plotscatter(Xmat,Ymat,a,b,plt):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(Xmat,Ymat,c='blue',marker='o')
    Xmat.sort()
    yhat=[a*float(xi)+b for xi in Xmat]
    plt.plot(Xmat,yhat,'r')
    plt.show()
    return yhat

def calculate1(Xmat,Ymat):
    meanx=mean(Xmat)
    meany=mean(Ymat)
    dx=Xmat-meanx
    dy=Ymat-meany
    sumxy=vdot(dx,dy)
    sqx=sum(power(dx,2))
    a=sumxy/sqx
    b=meany-a*meanx
    print a,b
    return a,b

def calculate2(Xmat,Ymat):
    meanx=mean(Xmat)
    meany=mean(Ymat)
    n=len(Xmat)
    amolecular=n*meanx*meany-sum(vdot(Xmat,Ymat))
    adenominator=n*power(meanx,2)-sum(vdot(Xmat,Xmat))
    a=amolecular/adenominator
    b=meany-a*meanx
    print a,b
    return a,b

def normalEquations(Xmat,Ymat):
    xArr=Xmat
    yArr=Ymat
    m=len(Xmat)
    Xmat=mat(ones((m,2)))
    for i in xrange(m):Xmat[i,1]=xArr[i]
    Ymat=mat(yArr).T
    xTx=Xmat.T*Xmat
    ws=[]
    if linalg.det(xTx)!=0.0:
        ws=xTx.I*(Xmat.T*Ymat)
    else:
        print '矩阵为奇异阵，无逆矩阵'
        sys.exit(0)
    print 'ws:',ws
    plotscatter(Xmat[:,1], Ymat, ws[1,0], ws[0,0], plt)
    

if __name__ == '__main__':
    Xmat,Ymat=loadDataSet(u'最小二乘法数据集.txt')
#     a1,b1=calculate1(Xmat, Ymat)

    #按照公式实现求解系数值
#     a2,b2=calculate2(Xmat, Ymat)
#     plotscatter(Xmat, Ymat, a1, b1, plt)
#     plotscatter(Xmat, Ymat, a2, b2, plt)

    #正规方程组的代码实现
    normalEquations(Xmat,Ymat)

'''
三种方法均得到正确的结果
'''









