#coding=UTF-8
'''
Created on 2015年12月28日

@author: 15361
'''
from numpy import *
import matplotlib.pyplot as plt
import sys
import os
from numpy.linalg.linalg import pinv

def plotscatter(Xmat,Ymat,yHat,plt):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(Xmat,Ymat,c='blue',marker='o')
    plt.plot(Xmat,yHat,'r')
    plt.show()
def loadDataSet(filename):
    numfeat=len(open(filename).readline().split('\t'))-1
    X=[];Y=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        X.append([float(curline[i]) for i in xrange(numfeat)])
        Y.append(float(curline[-1]))
    return X,Y

if __name__ == '__main__':
    xArr,yArr=loadDataSet(u'RBF数据集.txt') #数据矩阵，分类标签
    #RBF函数的平滑系数
    miu=0.02
    k=0.03
    #数据集坐标数组转换为矩阵
    xmat=mat(xArr);ymat=mat(yArr).T
    testArr=xArr #测试数组
    m,n=shape(xArr) #xArr的行数
    yHat=zeros(m) #yHat为y的预测值，yHat的数据是y的回归线矩阵
    for i in xrange(m):
        weights=mat(eye(m)) #权重矩阵
        for j in xrange(m):
            diffmat=testArr[i]-xmat[j,:]
            #利用RBF函数计算权重矩阵，计算后的权重是一个对角阵
            weights[j,j]=exp(diffmat*diffmat.T/(-miu*k**2))
        
        xTx=xmat.T*(weights*xmat) #矩阵左乘自身的转置
        if linalg.det(xTx)!=0.0:
            ws=xTx.I*(xmat.T*(weights*ymat))
            yHat[i]=testArr[i]*ws #计算回归线坐标矩阵
        else:
            print 'this matix is singular,canot do iinverse'
            sys.exit(0) #退出程序
    print corrcoef(yHat,ymat.T)#计算相关系数
    plotscatter(xmat[:,1], ymat, yHat, plt) #绘制图形
