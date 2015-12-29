#coding=utf-8
'''
Created on 2015年12月29日
@author: 15361
@link http://www.rueckstiess.net/research/snippets/show/72d2363e
'''
from scipy import *
from scipy.linalg import norm, pinv
 
from matplotlib import pyplot as plt
 
class RBF:
     
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in xrange(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
    
    def _basisfunc(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
    
    def _calcAct(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.numCenters), float)
        print 'shape(G)=',shape(G)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self._basisfunc(c, x)
        return G
     
    def train(self, X, Y):
        """ X: matrix of dimensions n x indim 
            y: column vector of dimension n x 1 """
         
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        
        print "center", self.centers
        # calculate activations of RBFs
        G = self._calcAct(X)
        print 'G=',G
        
        # calculate output weights (pseudoinverse)
        #pinv(G)该函数求矩阵的广义逆矩阵
        self.W = dot(pinv(G), Y)
         
    def test(self, X):
        """ X: matrix of dimensions n x indim """
        G = self._calcAct(X)
        Y = dot(G, self.W)
        return Y

def loadDataSet(filename):
    numfeat=len(open(filename).readline().split('\t'))-1
    X=[];Y=[]
    fr=open(filename)
    for line in fr.readlines():
        curline=line.strip().split('\t')
        X.append([float(curline[1])])
        Y.append(float(curline[-1]))
    return X,Y
    
if __name__ == '__main__':
    # ----- 1D Example ------------------------------------------------
    n = 100
    
    x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
    # set y and add random noise
    y = sin(3*(x+0.5)**3 - 1)
    # y += random.normal(0, 0.1, y.shape)
    # rbf regression
    rbf = RBF(1, 100, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')#真实值
    
    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)
    print corrcoef(y.T,z.T)
    #输出的相关系数
#     [[ 1.          0.99999964]
#      [ 0.99999964  1.        ]]
    
    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')
     
    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c-0.7, c+0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
     
    plt.xlim(-1.2, 1.2)
    plt.show()
