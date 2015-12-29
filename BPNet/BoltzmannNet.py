#coding=UTF-8
'''
Created on 2015年12月26日

@author: 15361
模拟退火算法
模拟退火的基本思想:
来源地址：http://ist.csu.edu.cn/ai/Ai/chapter3/351.htm
　　(1) 初始化：初始温度T(充分大)，初始解状态S(是算法迭代的起点)， 每个T值的迭代次数L
　　(2) 对k=1，……，L做第(3)至第6步：
 　　(3) 产生新解S′
　　(4) 计算增量Δt′=C(S′)-C(S)，其中C(S)为评价函数
 　　(5) 若Δt′<0则接受S′作为新的当前解，否则以概率exp(-Δt′/T)接受S′作为新的当前解.
　　(6) 如果满足终止条件则输出当前解作为最优解，结束程序。
 终止条件通常取为连续若干个新解都没有被接受时终止算法。
 　　(7) T逐渐减少，且T->0，然后转第2步
'''
from numpy import *
import matplotlib.pyplot as plt
import copy

class BoltzmannNet(object):
    def __init__(self):
        self.dataMat=[] # 外部导入的数据集
        self.MAX_ITER=2000 # 外循环迭代次数
        self.T0=1000 # 最高温度：这个温度的变化范围应位于最大迭代范围之内
        self.Lambda=0.97 # 温度下降参数
        self.iteration=0 # 达到最优势的迭代次数
        self.dist=[] # 存储生成的距离
        self.pathindx=[] # 存储生成的路径索引
        self.bestdist=0 # 生成的最优路径长度
        self.bestpath=[] # 生成的最优路径
        
    def loadDataSet(self,filename):
        fr=open(filename)
        for line in fr.readlines():
            lineArr=[]
            curLine=line.strip().split(',')
            lineArr.append(float(curLine[0]))
            lineArr.append(float(curLine[1]))
            self.dataMat.append(lineArr)
        self.dataMat=mat(self.dataMat)
    
    #矩阵各项量之间的欧氏距离
    def distEclud(self,matA,matB):
        distMat=[]
        for row in range(shape(matA)[0]):
            dd=[]
            for col in range(shape(matB)[1]):
                df =sum(power(matA[row,:]-matB[:,col].T,2))**0.5
                dd.append(df)
            distMat.append(dd)
        return mat(distMat)
    
    #波尔茨曼函数
    def boltzmann(self,newl,oldl,T):
        return exp(-(newl-oldl)/T)
    
    #计算路径长度
    def pathLen(self,dist,path):
        N=len(path)
        plen=0
        for i in xrange(0,N-1): #长度为N的向量，包含1——N的整数
            plen +=dist[path[i],path[i+1]]
        plen+=dist[path[0],path[N-1]]
        return plen
    #交换新旧路径
    def changePath(self,old_path):
        N=len(old_path)
        if random.rand()<0.25: #随机产生两个位置，并交换
            #floor函数向下取整        chpos保存两个元素的数组
            chpos=floor(random.rand(1,2)*N).tolist()[0]
            new_path=copy.deepcopy(old_path)
            new_path[int(chpos[0])]=old_path[int(chpos[1])]
            new_path[int(chpos[1])]=old_path[int(chpos[0])]
        else: #产生三个位置，交换a-b和b-c段的路径
            #ceil向上取整
            d=ceil(random.rand(1,3)*N).tolist()[0] #随机路径排序
            d.sort()
            a=int(d[0])
            b=int(d[1])
            c=int(d[2])
            if a!=b and b!=c:
                new_path=copy.deepcopy(old_path)
                new_path[a:c-1]=old_path[b-1:c-1]+old_path[a:b-1]
            else:
                new_path=self.changePath(old_path)
        return new_path
    
    #绘制散点图
    def drawScatter(self,plt):
        px=(self.dataMat[:,0]).tolist()
        py=(self.dataMat[:,1]).tolist()
        #s代表散点图的点的大小
        plt.scatter(px,py,c='green',marker='o',s=50)
        i=65 #从65开始获取对应的字符，作为标签
        for x,y in zip(px,py):
            plt.annotate(str(chr(i)),xy=(x[0]+40,y[0]),color='black')
            i+=1
    #绘制趋势线
    def TrendLine(self,plt,color='b'):
        plt.plot(range(len(self.dist)),self.dist,color)
    
    #构造一个初始可行解
    def initBMNet(self,m,n,distMat):
        self.pathindx=range(m)#返回的是小于m的一个列表
        #将pathindx中元素的顺序进行混洗，随机打乱次序
        random.shuffle(self.pathindx) #随机生成每个路径
        #每个路径对应的距离，计算这个次序下对应的路径长度，保存在dist中
        self.dist.append(self.pathLen(distMat, self.pathindx))
        #温度T返回默认的温度值
        return self.T0,self.pathindx,m
    
    #主函数
    def train(self):
        file =open('gailv.txt','w')
        [m,n]=shape(self.dataMat)
        #转换为邻接矩阵（距离矩阵）
        distMat=self.distEclud(self.dataMat, self.dataMat.T)
        #T为当前温度。curpath为当前路径索引，MAX_M为内循环最大迭代次数
        [T,curpath,MAX_M]=self.initBMNet(m,n,distMat)
        step=0 #初始化外循环迭代
        while step<=self.MAX_ITER: #外循环
            m=0 #内循环计数器
            while m<=MAX_M: #内循环
                curdist=self.pathLen(distMat, curpath) #计算当前路径距离
                newpath=self.changePath(curpath) #交换产生新路径
                newdist=self.pathLen(distMat, newpath) #计算新路径距离
                #判断网络是否是一个局部稳态
                if(curdist>=newdist):
                    curpath=newpath
                    self.pathindx.append(curpath)
                    self.dist.append(newdist)
                    self.iteration+=1 #增加迭代次数
                else: #如果网络处于局部稳态，则执行波尔茨曼函数
                    file.write(str(self.boltzmann(newdist,curdist,T))+"\n")
                    if random.rand()<self.boltzmann(newdist,curdist,T):
                        curpath=newpath
                        self.pathindx.append(curpath)
                        self.dist.append(newdist)
                        self.iteration+=1 #增加迭代次数
                m+=1
            step+=1
            #这里的降温策略选用线性函数进行降温
            T=T*self.Lambda #降温，返回迭代，直至算法终止
        self.bestdist=min(self.dist)
        indxes=argmin(self.dist)
        self.bestpath=self.pathindx[indxes]
    
    #绘制路径
    def drawPath(self,Seq,plt,color='b'):
        m,n=shape(self.dataMat)
        px=(self.dataMat[Seq,0]).tolist()
        py=(self.dataMat[Seq,1]).tolist()
        px.append(px[0])
        py.append(py[0])
        #默认连线各个点
        plt.plot(px,py,color)

if __name__ == '__main__':
    bmNet=BoltzmannNet()
    bmNet.loadDataSet('boltzmannData.txt')
    bmNet.train()
    print '循环迭代',bmNet.iteration,'次'
    print '最优解：',bmNet.bestdist
    print '最佳路线：',bmNet.bestpath
    bmNet.drawScatter(plt)
    bmNet.drawPath(bmNet.bestpath,plt)
    plt.show()
    
    bmNet.TrendLine(plt) #绘制算法收敛曲线
    plt.show()
    
    
    '''
循环迭代 5564 次
最优解： 14073.5680957
最佳路线： [6, 12, 10, 11, 13, 14, 0, 22, 23, 24, 20, 21, 19, 18, 17, 2, 16, 15, 5, 4, 3, 7, 8, 9, 1]

循环迭代 5369 次
最优解： 13923.4783949
最佳路线： [16, 15, 3, 1, 7, 8, 9, 6, 4, 5, 10, 11, 12, 13, 14, 0, 22, 24, 23, 20, 21, 19, 18, 17, 2]

循环迭代 5467 次
最优解： 14467.5552292
最佳路线： [23, 18, 17, 2, 16, 15, 7, 8, 9, 1, 3, 4, 5, 6, 12, 10, 11, 13, 14, 0, 22, 24, 20, 21, 19]

循环迭代 5366 次
最优解： 13878.1992204
最佳路线： [8, 1, 4, 5, 6, 12, 10, 11, 13, 14, 0, 24, 22, 23, 20, 21, 19, 18, 17, 2, 16, 15, 3, 7, 9]

循环迭代 5531 次
最优解： 14001.0611245
最佳路线： [17, 16, 18, 19, 21, 20, 23, 24, 0, 14, 13, 11, 22, 10, 12, 6, 5, 4, 15, 3, 1, 9, 8, 7, 2]

循环迭代 5295 次
最优解： 14274.3903632
最佳路线： [17, 16, 15, 3, 1, 4, 5, 10, 18, 19, 21, 20, 23, 24, 22, 0, 14, 13, 11, 12, 6, 9, 8, 7, 2]

    '''
    
    