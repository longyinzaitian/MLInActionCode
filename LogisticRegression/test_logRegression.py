#coding=utf-8
'''
Created on 2015年12月15日
@author: 15361
'''
#################################################
# logRegression: Logistic Regression
# Date   : 2014-03-02
# HomePage : http://blog.csdn.net/zouxy09
#################################################

from numpy import *
import matplotlib.pyplot as plt
from time import time
from logRegression import trainLogRegres,testLogRegres,showLogRegres

def loadData():
    train_x = []
    train_y = []
    fileIn = open('data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split()
        train_x.append([1.0, float(lineArr[0]), float(lineArr[1])])
        train_y.append(float(lineArr[2]))
    return mat(train_x), mat(train_y).transpose()

if __name__ == '__main__':
    ## step 1: load data
    print "step 1: load data..."
    train_x, train_y = loadData()
    test_x = train_x; test_y = train_y
    
    ## step 2: training...
    print "step 2: training..."
    opts = {'alpha': 0.01, 'maxIter': 20, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights = trainLogRegres(train_x, train_y, opts)
    print 'optimalWeights=',optimalWeights
    ## step 3: testing
    print "step 3: testing..."
    accuracy = testLogRegres(optimalWeights, test_x, test_y)
    print 'accuracy=',accuracy
    
    ## step 4: show the result
    print "step 4: show the result..."    
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    showLogRegres(optimalWeights, train_x, train_y) 
    
    
'''
梯度下降法，迭代500次
达到精度95%

随机梯度下降法 迭代200次
达到精度97%

改进的随机梯度下降法迭代200次
达到精度94%

改进的随机梯度下降法迭代20次
达到精度95%
'''
