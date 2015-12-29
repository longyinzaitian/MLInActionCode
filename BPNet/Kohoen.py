#_*_coding=UTF-8_*_
'''
Created on 2015年12月25日
参看维基百科算法步骤：
https://en.wikipedia.org/wiki/Self-organizing_map
1、Randomize the map's nodes' weight vectors
2、Grab an input vector D(t)
3、Traverse each node in the map
    Use the Euclidean distance formula to find the similarity
    between the input vector and the map's node's weight vector
    
    Track the node that produces the smallest distance
    (this node is the best matching unit, BMU)
4、Update the nodes in the neighborhood of the BMU (including the BMU itself)
    by pulling them closer to the input vector
    Wv(s + 1) = Wv(s) + Θ(u, v, s) α(s)(D(t) - Wv(s))
5、Increase s and repeat from step 2 while s < lambda
'''

import matplotlib.pyplot as plt
from numpy import *

class Kohoen(object):
    def __init__(self):
        self.lratemax=0.8   #最大学习率 欧氏距离
        self.lratemin=0.05   #最小学习率-欧氏距离
        self.rmax=5.0   #最大聚类半径 -根据数据集
        self.rmin=0.5   #最小聚类半径-根据数据集
        self.Steps=1000   #迭代次数
        self.lratelist=[]   #学习率收敛曲线
        self.rlist=[]   #学习半径收敛曲线
        self.w=[]   #权重向量组
        self.M=2   #M*N表示聚类总数
        self.N=2   #M N表示邻域的参数
        self.dataMat=[]   #外部导入数据集
        self.classLabel=[]   #聚类后的类别标签
        
    #归一化矩阵（按列进行归一化）
    def normalize(self,dataMat):
        [m,n]=shape(dataMat)
        for i in xrange(n):
            dataMat[:,i]=(dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
        return dataMat
    
    #计算矩阵欧式距离
    def distEclud(self,matA,matB):
        dd=[]
        for col in range(shape(matB)[1]):
            df =sum(power(matA-matB[:,col].T,2))**0.5
            dd.append(df)
        return mat(dd)
    
    def loadDataSet(self,fileName):
        numFeat=len(open(fileName).readline().split('\t'))-1
        fr=open(fileName)
        for line in fr.readlines():
            lineArr=[]
            curLine=line.strip().split('\t')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat=mat(self.dataMat)
    
    #初始化第二层网格
    def init_grid(self):
        k=0
        grid=mat(zeros((self.M*self.N,2)))
        for i in xrange(self.M):
            for j in xrange(self.N):
                grid[k,:]=[i,j]
                k+=1
        return grid
    
    #更新学习率和聚类半径
    def ratecalc(self,i):
        lrate =self.lratemax-(i+1.0)*(self.lratemax-self.lratemin)/(self.Steps)
        r=self.rmax-(i+1.0)*(self.rmax-self.rmin)/self.Steps
        return lrate,r
    
    def traint(self):
        dm,dn=shape(self.dataMat)#1、构建输入层网络
        normDataSet=self.normalize(self.dataMat) #归一化数据x
        grid=self.init_grid()#2.初始化第二层分类网络
        self.w=random.rand(dn,self.M*self.N)#3.随机初始化两层之间的权重网络
        distM=self.distEclud #确定距离公式
        #4.迭代求解
        if self.Steps<5*dm:#设定最小迭代次数
            self.Steps=5*dm
        for i in xrange(self.Steps):
            lrate,r=self.ratecalc(i)    #1）计算当前迭代次数下的学习率和分类半径
            self.lratelist.append(lrate)
            self.rlist.append(r)
            #2）随机生成样本索引，并抽取一个样本
            k=random.randint(0,dm)
            mySample=normDataSet[k,:]
            #3）计算最优节点，返回最小距离的索引值
            minIndx=(distM(mySample,self.w)).argmin()
            #4）计算邻域
            d1=ceil(minIndx/self.M) #计算此节点在第二层矩阵中的位置
            d2=mod(minIndx,self.M)
            distMat=distM(mat([d1,d2]),grid.T)
            nodelindx=(distMat<r).nonzero()[1] #获取邻域内的所有节点
            for j in xrange(shape(self.w)[1]): #按列更新权重
                if sum(nodelindx==j):
                    self.w[:,j]=self.w[:,j]+lrate*(mySample[0]-self.w[:,j])
        #主循环结束
        self.classLabels=range(dm) #分配和存储聚类后的类别标签
        for i in xrange(dm):
            self.classLabels[i]=distM(normDataSet[i,:],self.w).argmin()
        self.classLabels=mat(self.classLabels)

    def showCluster(self,plt):
        lst=unique(self.classLabels.tolist()[0])
        i=0
        for cindx in lst:
            myclass=nonzero(self.classLabels==cindx)[1]
            xx=self.dataMat[myclass].copy()
            if i==0:
                plt.plot(xx[:,0],xx[:,1],'bo')
            elif i==1:
                plt.plot(xx[:,0],xx[:,1],'rd')
            elif i==2:
                plt.plot(xx[:,0],xx[:,1],'gD')
            elif i==3:
                plt.plot(xx[:,0],xx[:,1],'c^')
            elif i==4:
                plt.plot(xx[:,0],xx[:,1],'mH')
            elif i==5:
                plt.plot(xx[:,0],xx[:,1],'kh')
            i+=1
        plt.show()

if __name__ == '__main__':
    somnet=Kohoen()
    somnet.loadDataSet('SOMnetData.txt')
    somnet.traint()
    print somnet.w
    somnet.showCluster(plt)
            