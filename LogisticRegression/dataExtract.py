#coding=utf-8
'''
Created on 2015年12月15日

@author: 15361
'''
import time
from datetime import datetime
from numpy import *
from logRegression import testLogRegres,trainLogRegres,showLogRegres

def timeconver(t):
    dd =datetime.strptime(t,"%Y-%m-%dT%H:%M:%S.%fZ")
    return time.mktime(dd.timetuple())

def timegap(t2,t1):
    return timeconver(t2)-timeconver(t1)

def generateFile():
    clickP = u'G://下载的论文//推荐系统论文//DataSets//yoochoose-dataFull//clicktest.dat'
    buyP=u'G://下载的论文//推荐系统论文//DataSets//yoochoose-dataFull//buytest.dat'
    extract = u'G://下载的论文//推荐系统论文//DataSets//yoochoose-dataFull//clickbuy.dat'
    clickInfo = open(clickP,'r')
    buyInfo = open(buyP,'r')
    clickmap = {}
    for clickline in clickInfo:
        clickitem = clickline.strip().split(',')
        if clickmap.get(clickitem[0]) is None:
            clickmap.setdefault(clickitem[0],[])
            clickmap.get(clickitem[0]).append((clickitem[1],clickitem[2],clickitem[3]))
        else:
            clickmap.get(clickitem[0]).append((clickitem[1],clickitem[2],clickitem[3]))
    buymap = {}
    for buyline in buyInfo:
        buyitem = buyline.strip().split(',')
        if buymap.get(buyitem[0]) is None:
            buymap.setdefault(buyitem[0],[])
            buymap.get(buyitem[0]).append((buyitem[1],buyitem[2],buyitem[3]))
        else:
            buymap.get(buyitem[0]).append((buyitem[1],buyitem[2],buyitem[3]))
    
    extractfile = open(extract,'w')
    extractMap ={}
    for clicki in clickmap:
        bi = buymap.get(clicki)
        ci = clickmap.get(clicki)
        if extractMap.get(clicki) is None:
            #计算时间差
            td=timegap(ci[-1][0].__str__(),ci[0][0].__str__())
            if bi is not None:#如果购买过
                extractMap.setdefault(clicki,(td,len(ci),1))
            else:#没有购买过
                extractMap.setdefault(clicki,(td,len(ci),0))
    
    for item in extractMap:
        ii = extractMap.get(item)
        #sid,time,clicknum,yesno
        extractfile.write(item+','+ii[0].__repr__()+','+ii[1].__repr__()+','+ii[2].__repr__()+'\n')
#         if bi is None:
#             td=timegap(ci[-1][0].__str__(),ci[0][0].__str__())
#             extractfile.write(clicki+','+td.__repr__()+','+0.0.__repr__()+','+len(ci).__str__()+',0\n')
#         else:
#             td=timegap(bi[-1][0], ci[0][0])
#             extractfile.write(clicki+','+td.__repr__()+','+bi[0][1]+','+(len(ci)+len(bi)).__repr__()+','+'1\n')

def loadData():
    train_x = []
    train_y = []
    fileIn = open(u'G://下载的论文//推荐系统论文//DataSets//yoochoose-dataFull//clickbuy.dat')
    for line in fileIn.readlines():
        lineArr = line.strip().split(',')
        train_x.append([1.0, float(lineArr[1]), float(lineArr[2])])
        train_y.append(float(lineArr[3]))
    x = mat(train_x)
    mx = x.max(0)
    print mx.getA()
    print 'x.max=',mx[0,1]
    mmx =mx[0,1]/10.0
    print mmx
    for tx in train_x:
        tx[1]/=mmx
    return mat(train_x), mat(train_y).transpose()

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

if __name__ == '__main__':
#     generateFile()
    
    ## step 1: load data
    print "step 1: load data..."
    train_x, train_y = loadData()
    
    test_x = train_x; test_y = train_y
    
    ## step 2: training...
    print "step 2: training..."
    opts = {'alpha': 0.01, 'maxIter': 200, 'optimizeType': 'smoothStocGradDescent'}
    optimalWeights = trainLogRegres(train_x, train_y, opts)
    print 'optimalWeights=',optimalWeights
    ## step 3: testing
    print "step 3: testing..."
    accuracy = testLogRegres(optimalWeights, test_x, test_y)
    
#     numSamples, numFeatures = shape(test_x)
#     matchCount = 0
#     for i in xrange(numSamples):
#         predict = sigmoid(test_x[i, :] * optimalWeights)[0, 0] > 0.5
#         if predict == bool(test_y[i, 0]):
#             matchCount += 1
#     accuracy = float(matchCount) / numSamples
    
    print 'accuracy=',accuracy
    
    ## step 4: show the result
    print "step 4: show the result..."    
    print 'The classify accuracy is: %.3f%%' % (accuracy * 100)
    showLogRegres(optimalWeights, train_x, train_y) 
    '''
    #运算结果很精确
    Congratulations, training complete! Took 0.666000s!
    optimalWeights= [[-11.70491496]
    [ -3.65040344]
    [  2.31317572]]
    step 3: testing...
    accuracy= 0.964285714286
    step 4: show the result...
    The classify accuracy is: 96.429%
    '''