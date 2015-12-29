#coding=utf-8
'''
Created on 2015年12月23日
@author: 15361
BP神经网络
此代码建议与以下网址结合理解：
http://python.jobbole.com/82758/
该网址中的内容更加贴切BP神经网络的推导过程
'''

from numpy import *
import matplotlib.pyplot as plt
import operator
class BPNet(object):
    #构造函数
    def __init__(self):
        self.eb =0.01 #误差容限
        self.iterator=0 #算法收敛时的迭代次数
        self.eta=0.1 #学习率
        self.mc=0.3 #动量因子
        self.maxiter=5000 #最大迭代次数
        self.nHidden=4 #隐含层神经元
        self.nOut=1 #输出层个数
        #以下属性由系统生成
        self.errlist=[] #误差列表
        self.dataMat=[] #训练集
        self.classLabels=0 #分类标签机
        self.nSampNum=0 #样本集行数
        self.nSampDim=0 #样本列数
        pass
    #激活（传递）函数
    def logistic(self,net):
        return 1.0/(1.0+exp(-net))
    #激活函数的导函数
    def dlogit(self,net):
        return multiply(net,(1.0-net))
    #矩阵个元素平方之和
    def errorfunc(self,inX):
        return sum(power(inX,2))*0.5
    #数据标准归一化
    def normalize(self,dataMat):
        [m,n]=shape(dataMat)
        for i in xrange(n-1):
            dataMat[:,i]=(dataMat[:,i]-mean(dataMat[:,i]))/(std(dataMat[:,i])+1.0e-10)
        return dataMat
    #加载数据
    def loadDataSet(self,filename):
        self.dataMat=[]
        self.classLabels=[]
        fr=open(filename)
        for line in fr.readlines():
            lineArr=line.strip().split()
            self.dataMat.append([float(lineArr[0]),float(lineArr[1]),1.0])
            self.classLabels.append(int(lineArr[2]))
            
        self.dataMat=mat(self.dataMat)
        m,n=shape(self.dataMat)
        self.nSampNum=m #样本数量
        self.nSampDim=n-1 #样本维度
    #增加新列
    def addcol(self,matrix1,matrix2):
        [m1,n1]=shape(matrix1)
        [m2,n2]=shape(matrix2)
        if m1!=m2:
            print 'different rows,can not merge matrix'
            return ;
        mergMat = zeros((m1,n1+n2))
        mergMat[:,0:n1]=matrix1[:,0:n1]
        mergMat[:,n1:n1+n2]=matrix2[:,0:n2]
        return mergMat
    #隐含层初始化
    def init_hiddenWB(self):
        self.hi_w=2.0*(random.rand(self.nHidden,self.nSampDim)-0.5) #(4,n-1)
        self.hi_b=2.0*(random.rand(self.nHidden,1)-0.5) #(4,1)
        self.hi_wb=mat(self.addcol(mat(self.hi_w), mat(self.hi_b))) #(4,n)
    #输出层初始化
    def init_OutputWB(self):
        self.out_w=2.0*(random.rand(self.nOut,self.nHidden)-0.5)#(1,4)
        self.out_b=2.0*(random.rand(self.nOut,1)-0.5)#(1,1)
        self.out_wb=mat(self.addcol(mat(self.out_w), mat(self.out_b))) #(1,5)
    #BP网络主程序
    def bpTrain(self):
        SampleIn=self.dataMat.T #(n m)
        expected=mat(self.classLabels) #(1,m)
        self.init_hiddenWB()
        self.init_OutputWB()
        dout_wbOld=0.0 #默认t-1权值
        dhi_wbOld=0.0
        for i in xrange(self.maxiter):
            #1.工作信号正向传播
            #1.1信息从输入层到隐含层：这里使用了矢量计算，计算的是整个样本集的结果。结果是4行307列矩阵
            hi_input=self.hi_wb*SampleIn #(4,n)*(m,n)=(4,n)
            hi_output=self.logistic(hi_input) #(4,n)
            hi2out=self.addcol(hi_output.T, ones((self.nSampNum,1))).T #(n,4)(n,1)=(n,5).T=(5,n)
            #1.2从隐含层到输出层：结果是5行307列的矩阵
            out_input=self.out_wb*hi2out #(1,5)*(n,5) =(1,5)
            out_output=self.logistic(out_input) #(1,5)
            
            #2 误差计算
            err=expected-out_output #(1,5)
            sse=self.errorfunc(err)
            self.errlist.append(sse)
            #判断是否收敛至最优
            if sse<=self.eb:
                self.iterator = i+1
                break;
            
            #3 误差信号反向传播
            #multiply与*
            #这两种方式进行的矩阵相乘并没有什么本质的区别。可以相互进行转化。
            DELTA=multiply(err,self.dlogit(out_output)) #DELTA为输出层梯度 (1,5) (1,5) =(1,5)
            
            #delta为隐含层梯度
            delta=multiply(self.out_wb[:,:-1].T*DELTA,self.dlogit(hi_output))
            dout_wb=DELTA*hi2out.T  #输出层权值微分
            dhi_wb=delta*SampleIn.T #隐含层权值微分
            if i==0:
                self.out_wb=self.out_wb+self.eta*dout_wb
                self.hi_wb=self.hi_wb+self.eta*dhi_wb
            else:
                self.out_wb=self.out_wb + (1.0-self.mc)*self.eta*dout_wb+self.mc*dout_wbOld
                self.hi_wb=self.hi_wb + (1.0-self.mc)*self.eta*dhi_wb+self.mc*dhi_wbOld
                dout_wbOld= dout_wb
                dhi_wbOld=dhi_wb

    #BP网络分类器
    def BPClassfier(self,start,end,steps=30):
        x=linspace(start,end,steps)
        xx=mat(ones((steps,steps)))
        xx[:,0:steps]=x
        yy=xx.T
        z=ones((len(xx),len(yy)))
        for i in range(len(xx)):
            for j in range(len(yy)):
                xi=[]
                taues=[]
                tautemp=[]
                mat(xi.append([xx[i,j],yy[i,j],1]))
                hi_input=self.hi_wb*(mat(xi).T)
                hi_out=self.logistic(hi_input)
                taumrow,taucol=shape(hi_out)
                tauex=mat(ones((1,taumrow+1)))
                tauex[:,0:taumrow]=(hi_out.T)[:,0:taumrow]
                out_input=self.out_wb*(mat(tauex).T)
                out=self.logistic(out_input)
                z[i,j]=out
        return x,z
            
    #绘制分类线
    def classfyLine(self,plt,x,z):
        plt.contour(x,x,z,1,colors='black')
    #绘制趋势线：可调整颜色
    def TrendLine(self,plt,color='r'):
        x=linspace(0,self.maxiter,self.maxiter)
        y=log2(self.errlist)
        plt.plot(x,y,color)
    #绘制分类点
    def drawClassScatter(self,plt):
        i=0
        for mydata in self.dataMat:
            if self.classLabels[i]==0:
                plt.scatter(mydata[0,0],mydata[0,1],c='blue',marker='o')
            else:
                plt.scatter(mydata[0,0],mydata[0,1],c='red',marker='s')
            i+=1

if __name__ == '__main__':
    bpnet=BPNet()
    bpnet.loadDataSet('BPNetData.txt')
    bpnet.dataMat=bpnet.normalize(bpnet.dataMat)
    #绘制数据集散点图
    bpnet.drawClassScatter(plt)
    #BP神经网络进行数据分类
    bpnet.bpTrain()
    
    print bpnet.out_wb
    print bpnet.hi_wb
    #计算和绘制分类线
    x,z=bpnet.BPClassfier(-3.0, 3.0)
    bpnet.classfyLine(plt, x, z)
    plt.show()
    #绘制误差曲线
    bpnet.TrendLine(plt)
    plt.show()
    
    
    
    