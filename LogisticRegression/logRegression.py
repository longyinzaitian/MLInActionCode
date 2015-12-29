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

import time
from numpy import *
import matplotlib.pyplot as plt

# calculate the sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

# train a logistic regression model using some optional optimize algorithm
# input: train_x is a mat datatype, each row stands for one sample
#         train_y is mat datatype too, each row is the corresponding label
#         opts is optimize option include step and maximum number of iterations
def trainLogRegres(train_x, train_y, opts):
    # calculate training time
    startTime = time.time()

    numSamples, numFeatures = shape(train_x)
    print 'numSaples=',numSamples
    print 'numFeatures=',numFeatures
    alpha = opts['alpha']; maxIter = opts['maxIter']
    weights = ones((numFeatures, 1))#三行一列
    print 'weights=',weights
    # optimize through gradient descent algorilthm
    for k in range(maxIter):
        #梯度下降算法
        if opts['optimizeType'] == 'gradDescent': # gradient descent algorilthm
            output = sigmoid(train_x * weights)#输出100行一列矩阵
            error = train_y - output#输出100行一列矩阵
            weights = weights + alpha * train_x.transpose() * error#三行一列
            #随机梯度下降算法
        elif opts['optimizeType'] == 'stocGradDescent': # stochastic gradient descent
            for i in range(numSamples):
                output = sigmoid(train_x[i, :] * weights)
                error = train_y[i, 0] - output
                weights = weights + alpha * train_x[i, :].transpose() * error
            #改进的随机梯度下降算法
        elif opts['optimizeType'] == 'smoothStocGradDescent': # smooth stochastic gradient descent
            # randomly select samples to optimize for reducing cycle fluctuations 
            dataIndex = range(numSamples)
            for i in range(numSamples):
                alpha = 4.0 / (1.0 + k + i) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                output = sigmoid(train_x[randIndex, :] * weights)
                error = train_y[randIndex, 0] - output
                weights = weights + alpha * train_x[randIndex, :].transpose() * error
                del(dataIndex[randIndex]) # during one interation, delete the optimized sample
        else:
            raise NameError('Not support optimize method type!')
    
    print 'Congratulations, training complete! Took %fs!' % (time.time() - startTime)
    return weights

# test your trained Logistic Regression model given test set
def testLogRegres(weights, test_x, test_y):
    numSamples, numFeatures = shape(test_x)
    matchCount = 0
    for i in xrange(numSamples):
        predict = sigmoid(test_x[i, :] * weights)[0, 0] > 0.5
        if predict == bool(test_y[i, 0]):
            matchCount += 1
    accuracy = float(matchCount) / numSamples
    return accuracy


# show your trained logistic regression model only available with 2-D data
def showLogRegres(weights, train_x, train_y):
    # notice: train_x and train_y is mat datatype
    numSamples, numFeatures = shape(train_x)
    if numFeatures != 3:
        print "Sorry! I can not draw because the dimension of your data is not 2!"
        return 1

    # draw all samples
    for i in xrange(numSamples):
        if int(train_y[i, 0]) == 0:
            plt.plot(train_x[i, 1], train_x[i, 2], 'or')
        elif int(train_y[i, 0]) == 1:
            plt.plot(train_x[i, 1], train_x[i, 2], 'ob')

    # draw the classify line
    min_x = min(train_x[:, 1])[0, 0]
    max_x = max(train_x[:, 1])[0, 0]
    weights = weights.getA()  # convert mat to array
    #此处得出该公式是运用线性模型得到的。
    y_min_x = float(-weights[0] - weights[1] * min_x) / weights[2]
    y_max_x = float(-weights[0] - weights[1] * max_x) / weights[2]
    plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()