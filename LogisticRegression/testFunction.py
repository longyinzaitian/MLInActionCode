#coding=utf-8
'''
Created on 2015年12月15日

@author: 15361
'''
from numpy import *
import time
from datetime import datetime,date
# from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.pyplot import contour
from scipy.linalg.misc import norm
if __name__ == '__main__':
    '''
    #矩阵有一个shape属性，是一个(行，列)形式的元组
    a = array([[1,2,3],[4,5,6]])
    print a.shape
    '''
    '''
    #返回按要求的矩阵
    ones = ones((2,1))
    print ones
    '''
    #计算结果也是矩阵
    '''
    from logRegression import sigmoid
    from numpy import mat
    aa = mat([1,2,3,4])
    bb = sigmoid(aa)
    print bb
    '''
    '''
    for i in xrange(3):
        print i
    test=[1,2,3,4]
    print test[:]
    print test[2:3]
    for i in xrange(2,5):
        print i
    '''
    
    '''
    dd = datetime.strptime('2014-04-03T10:53:49.875Z', "%Y-%m-%dT%H:%M:%S.%fZ")
    print time.mktime(dd.timetuple())#1396493629.0
    '''
    '''
    tuple =(1,2,3)
    print tuple[len(tuple)-1]
    print tuple[-1]
    print 9.99.__repr__()
    print str(9.99)
    print tuple[-2]
    '''
    '''
    aa =[[1],[2],[3]]
    aa= mat(aa)#将列表转换成矩阵，并存放在aa中
    print aa
    print aa.transpose()#将矩阵进行转置
    print aa #transpose()进行矩阵的转置，aa并没有改变
    print '*'*20
    print aa.T
    print aa #T转置矩阵，也没有发生改变
    '''
    '''
    print '*'*20
    print zeros((2,1))
    print ones((2,3))#参数指明了矩阵的行列数
    '''
    '''
    #列表和数组的区别：
    #列表： [1, 2, 3, 4]
    #数组： [1 2 3 4]
    print '*'*20
    ll =[1,2,3,4]
    print '列表：',ll
    arr = array(ll)
    print '数组：',arr
    
    print '*'*20
    print linspace(0,3,6) #返回的是数组
    '''
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xx = linspace(0,3,6)
    yy = xx*1.5
    #画点图
    ax.scatter(xx,xx,c='green',marker='s')
    ax.scatter(xx,yy,c='blue',marker='o')
#     画x与y的关系图
    ax.plot(xx,0.75+xx,'r')
    plt.show()
    '''
    '''
    print '**************数组排序问题****************'
    #数组的构建问题，初始化使用array()
    ary=array(zeros(4))
    ary[0]=0.1
    ary[1]= 0.6
    ary[2]= 0.5
    ary[3]= 0.7
    #有-号，降序排列
    #无-号，升序排列
    sortindex = argsort(-ary)
    for id in sortindex:
        print '索引：',id
    for i in sortindex:
        print ary[i]
    '''
    '''
    #矩阵元素的获取
    ll = [[1,2,3],[4,5,6],[7,8,9]]
    #获取第二行第0个元素
    print mat(ll)[2,0]
    #第一个冒号代表获取行的起止行号
    #第二个冒号代表获取列的起止行号
    print mat(ll)[:,:]
    '''
    '''
    #构建对角矩阵 
    #diag()参数为列表即可
    dd = [1,2,3]
    dilogg = diag(dd)
    print 'diag=',dilogg
    '''
    
    '''
    dd = [1,2,3]
    dilogg = diag(dd)
    print 'diag=',dilogg
    print 'dd:',linalg.inv(dilogg)
    print 'I=',mat(dilogg).I
    ll = [[1,2,3],[4,5,6],[7,8,9]]
    #求逆矩阵
    lv = linalg.inv(mat(ll))
    print 'inv:',lv
    print 'I:',mat(ll).I
    '''
    '''
    ll = [[1,2,3],[4,5,6],[7,8,9]]
    ld = dot(ll,ll)
    print 'dot:',ld
    print mat(ll)*mat(ll)
    '''
    '''
    print 'eye:',eye(2)#单元矩阵
    '''
    '''
    ll = [[1,2,3],[4,5,6],[7,8,9]]
    print 'allclose:',allclose(ll,eye(3))
    '''
    '''
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    U =A*A.T
    lamda,hU=linalg.eig(U)
    VT=A.T*A
    eV,hVT=linalg.eig(VT)
    hV=hVT.T
    print 'hU:',hU
    print 'hV:',hV
    print lamda
    #排序 降序排列
    sigma=sorted(sqrt(lamda),reverse=True)
    dd=diag(sigma)
    print 'diag=',dd
    #单位矩阵 4行5列
    ddr=mat(dd)*eye(4,5)
    print 'ddr=',ddr
    print '====',mat(hU)*(mat(diag(sigma))*eye(4,5))*(hVT)
    print '*'*20
    Sigma =zeros([shape(A)[0],shape(A)[1]])
    
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    U,S,VT =linalg.svd(A)
    print 'U:',U
    print 'V:',VT
    print 's:',S
    print '===',U*(mat(diag(S))*eye(4,5))*VT
    
    #测试逆矩阵
    print '*'*30
    ab =[[1,2],[3,4]]
    print linalg.inv(mat(ab))
    #[[-2.   1. ]
    #[ 1.5 -0.5]]
    '''
    '''
    ll=[8,0,3,6,1,0,5,3,8,9]
    print sorted(ll,reverse=True)
    print sorted(ll,reverse=False)
    '''
    '''
    ab =[[1,0],[0,4]]
    print mat(ab).I #求逆矩阵
    
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    U,S,VT =linalg.svd(A)
    print 'U:',U
    print 'V:',VT
    print 's:',S
    print '===',U*(mat(diag(S))*eye(4,5))*VT
    print '--'*30
    print 'u*ut',U*U.T
    print 'v.vt',VT*VT.T
    
    print '+'*30
    ptv=mat([[445],[455],[332],[454],[444],[354],[443],[244],[555]])
    U1,S1,VT1 =linalg.svd(ptv,full_matrices=False)
    print 'U:',U1
    print 'V:',VT1
    print 's:',S1
    '''
    '''
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    print A[:,1]
    
    #获取3*3个0-1之间的数字
    rr=random.rand(3,3)
    print rr
    print (rr-0.5)
    print 2.0*(rr-0.5)
    '''
    '''
    delta = 0.25
    x = arange(-3.0, 3.0, delta)
    print x
    '''
    '''
    delta = 0.025
    x = arange(-3.0, 3.0, delta)
    y = arange(-2.0, 2.0, delta) 
    xx,yy=meshgrid(x,y)
    z=power(xx, 2)+power(yy,2)
    plt.figure()
    #设置坐标极值
    plt.xlim(-3.5,3.5)
    plt.ylim(-2.5,2.5)
    #设置刻度值
    plt.xticks(linspace(-3.5,3.5,15,endpoint=True))
    plt.yticks(linspace(-2.5,2.5,11,endpoint=True))
    plt.contour(xx,yy,z)
    plt.show()
    '''
    '''
    x=linspace(-50,70,20)
    y=linspace(-40,40,20)
    xx,yy=meshgrid(x,y)
    z=mat(ones((20,20)))
    z=xx+yy
    plt.scatter(xx,yy,c='red',marker='s')
    z=z<zeros((20,20))
    for i in range(len(xx)):
        for j in range(len(yy)):
            pass
    plt.contour(xx,yy,z,2)
    plt.show()
    '''
    '''
    z=xx+yy
    m,n=shape(z)
    plt.figure()
    for i in range(m):
        for j in range(n):
            if z[i,j]>0:
                z[i,j]=1
                plt.scatter(xx[i],yy[j],c='blue',marker='o')
            else:
                z[i,j]=0
                plt.scatter(xx[i],yy[j],c='red',marker='s')
    
    plt.contour(xx,yy,z,2)
    plt.show()
    '''
    
    '''
    xx=linspace(-3.0,3.0,256)
    yy=linspace(-3.0,3.0,256)
    print xx
    print '*'*30
    x,y=meshgrid(xx,yy)
    z=(1 - x / 2 + x ** 5 + y ** 3) * exp(-x ** 2 -y ** 2)
    plt.figure()
    plt.contour(x,y,z,8,alpha=0.75,cmap='jet')
    plt.show()
    '''
    
    '''
    x =[[1,0,0,0,2],[0,0,3,0,0]]
    print x
    nz=nonzero(x)
    print nz
    print nz[0]
    
    #nz返回值
    # 第一行是所有非零数所在行值
    # 第二行是所有非零值所在列值
    '''
    '''
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    sample =A[0,:]
    print sample
    print sample[0]
    
    ll=mat([3,4,5])
    for i in range(5):
        if sum(ll==i):
            print i
    '''
    
    '''
    A=mat([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
    #根据ind序列索引获取矩阵A中的数据
    ind=[2,1,3,0]
    print A[ind,0]
    '''
    '''
    print random.rand()
    #打印出来的是数组
    print floor(random.rand(1,2)*10)[0]
    '''
    '''
    ll=[1,2,3,4,5,6]
    #可以互换指定区域的位置
    print ll[3:6]+ll[0:3]
    #成对获取x、y的值
    l1=[1,2,3]
    l2=[4,5,6]
    for x,y in zip(l1,l2):
        print x,y
    '''
    '''
    #获取指定的字符
    for i in range(65,70):
        print str(chr(i))
    '''
    '''
    px=[1,3,4,9]
    py=[3,4,6,0]
    plt.figure()
    plt.xlim(0,10)
    plt.ylim(-1,7)
    plt.plot(px,py,'b--')
    plt.show()
    '''
    '''
    ll=range(9)#返回列表
    print ll
    #shuffle函数随机打乱列表中的元素顺序
    print random.shuffle(ll)
    print ll
    '''
    
    '''
    #波尔茨曼函数 其实是一个指数函数
    def boltzmann(newl,oldl,T):
        return exp(-(newl-oldl)/T)
    x=linspace(0,20,200)
#     y1=1.0/(1.0+exp(-(x)/4))
#     y2=1.0/(1.0+exp(-(x)/1))
#     y3=1.0/(1.0+exp(-(x)/0.25))
    y1=(exp(-(x)/4))
    y2=(exp(-(x)/1))
    y3=(exp(-(x)/0.25))
#     for i in range(-20,20):
#         x.append(i)
#         y.append(boltzmann(i, 0, 4))
    plt.figure()
    plt.xlim(-5,25)
    
#     plt.plot(x,y1,'-',x,y2,'--',x,y3,'.')
    plt.plot(x,y1,'-')
    plt.show()
    '''
    
    '''
    #vdot 返回两向量的点积
    l1=[1,2,3]
    l2=[4,5,6]
    ll=[[1,2,3],[4,5,6],[1,2,3]]
    print vdot(l1,l2)
    print dot(l1,l2)
    
    print mat(l1)*mat(l2).T
    print mat(ll)
    
    print dot(mat(ll),mat(ll).T)
    print vdot(mat(ll),mat(ll))
    
    '''
    '''
    ll=[[-2.1,-1,4.3],[3,1.1,0.12]]
    print cov(ll)
    l=[-2.1,-1,4.3]
    lm=l-mean(l)
    print (lm*lm.T)/4
    print '=='*20
    vc=[1,2,39,0,8]
    vb=[1,2,38,0,8]
    va=[0,2,36,0,4]
    print cov(vc,vb)
    print mean(multiply(vc,vb))-mean(vc)*mean(vb)
    print mean(multiply((vc-mean(vc)),(vb-mean(vb))))
    print mean(multiply((vc-mean(vc)),(vb-mean(vb))))/(std(vb)*std(vc))
    #corrcoef得到相关系数矩阵（向量的相似程度）
    print corrcoef(vc,vb)
    
    print '*'*20
    x=[[0, 1, 2],[2, 1, 0]]
    print cov(x)
    print sum((multiply(x[0],x[1]))-mean(x[0])*mean(x[1]))/2
    print '*'*20
    b=[1,3,5,6]
    print cov(b)
    print sum((multiply(b,b))-mean(b)*mean(b))/3
    print '*'*20
    print sum(power(b-mean(b),2))/4
    
    x1=[1,2,3];x2=[4,5,6]
    print multiply(x1,x2)
    
    print var(b)
    print power(std(b),2)
    print mean(b)
    print sum(b)
    print sum(x)
    '''
    
    ################################################################################
    '''
    print 3*2**2
    print 3*2**0.5
    print (3*2)**2
    print (3*2)**0.5
    '''
    '''
    ll=[3,4,6,2,89,9,3,2]
    print max(ll)
    l2=[[3,4,6,2,89,9,3,2],[3,6,7,8,983,3,5,6]]
    print max(l2[0])
    print max(l2)
    '''
    
    '''
    #开始值，结束值，步长。如果步长为虚数，表示产生的个数长度
    print mgrid[-5:5:3j] #结果：[-5.  0.  5.]
    print mgrid[-5:5:3] #结果：[-5 -2  1  4]
    print '*'*20
    print mgrid[-5:5:3j,-5:5:3j]
    '''
    '''
    [[[-5. -5. -5.]
    [ 0.  0.  0.]
    [ 5.  5.  5.]]
    [[-5.  0.  5.]
    [-5.  0.  5.]
    [-5.  0.  5.]]]
    '''
    '''
    print mgrid[-5:5:3,-5:5:3]
    '''
    '''
    [[[-5 -5 -5 -5]
      [-2 -2 -2 -2]
      [ 1  1  1  1]
      [ 4  4  4  4]]

     [[-5 -2  1  4]
      [-5 -2  1  4]
      [-5 -2  1  4]
      [-5 -2  1  4]]]
    '''
    '''
    print '*'*40
    print ogrid[-5:5:3j]
    print ogrid[-5:5:3]
    print ogrid[-5:5:3j,-5:5:3j]
    print ogrid[-5:5:3,-5:5:3]
    '''
    '''
    [-5.  0.  5.]
    [-5 -2  1  4]
    [array([[-5.],
           [ 0.],
           [ 5.]]), array([[-5.,  0.,  5.]])]
    [array([[-5],
           [-2],
           [ 1],
           [ 4]]), array([[-5, -2,  1,  4]])]
    '''
    
    '''
    print random.seed(1)
    #要每次产生随机数相同就要设置种子，相同种子数的Random对象，相同次数生成的随机数字是完全相同的
    '''
    '''
    #用于生成一个指定范围内的随机符点数
    print random.uniform(-1,1,5)
    #结果：[ 0.40254497 -0.42350395 -0.67640645 -0.54075394 -0.99584028]
    #均匀分布
    '''
    '''
    #用于生成一个0到1的随机符点数: 0 <= n < 1.0
    print random.random()
    print random.random(5)
    print random.rand(2,3)#2行3列
    '''
    '''
    #用于生成一个指定范围内的整数。其中参数a是下限，参数b是上限，生成的随机数n: a <= n <= b
    print random.randint(5, 10)
    print random.randint(5,10,size=(5,5))
    '''
#以下内容来源博客：http://blog.csdn.net/pipisorry/article/details/39086463
    #random.randint(a, b, size=(c, d))
    #random.randrange([start], stop[, step])
    #random.choice(sequence)
    #random.shuffle(x[, random])
    #random.sample(sequence, k)
    
# linspace(start, end, num): 如linspace(0,1,11)结果为[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
# arange(n): 产生一个从0到n-1的向量，如arange(4)结果为[0,1,2,3]
# random.random([...]): 产生随机矩阵，如random.random([2,3])产生一个2x3维的随机数
# Simple random data
# rand(d0, d1, ..., dn)    Random values in a given shape.
# randn(d0, d1, ..., dn)    Return a sample (or samples) from the “standard normal” distribution.
# randint(low[, high, size])    Return random integers from low (inclusive) to high (exclusive).
# random_integers(low[, high, size])    Return random integers between low and high, inclusive.
# random_sample([size])    Return random floats in the half-open interval [0.0, 1.0).
# random([size])    Return random floats in the half-open interval [0.0, 1.0).
# ranf([size])    Return random floats in the half-open interval [0.0, 1.0).
# sample([size])    Return random floats in the half-open interval [0.0, 1.0).
# choice(a[, size, replace, p])    Generates a random sample from a given 1-D array ..
# bytes(length)    Return random bytes.
# random.expovariate(lambd) 指数分布

# random.gammavariate(alpha, beta)
# gamma分布
# 
# random.gauss(mu, sigma)
# 高斯分布
# 
# random.lognormvariate(mu, sigma)
# 对数正态分布

# random.normalvariate(mu, sigma)
# 正态分布
# 
# random.vonmisesvariate(mu, kappa)
# 冯·米塞斯分布（von Mises distribution）指一种圆上连续概率分布模型，它也被称作循环正态分布（circular normal distribution）
# 
# random.paretovariate(alpha)
# 帕累托分布是以意大利经济学家维弗雷多·帕雷托命名的。 是从大量真实世界的现象中发现的幂次定律分布。这个分布在经济学以外，也被称为布拉德福分布
# 
# random.weibullvariate(alpha, beta)
# 韦伯分布（Weibull distribution），又称韦氏分布或威布尔分布，是可靠性分析和寿命检验的理论基础

#以下内容来源博客：http://blog.csdn.net/pipisorry/article/details/39088003
#    tofile和fromfile数组内建函数（not recommend）
# 使用数组的方法函数tofile可以方便地将数组中数据以二进制的格式写进文件。tofile输出的数据没有格式，因此用numpy.fromfile读回来的时候需要自己格式化数据

# Note:
# 1. 读入的时候设置正确的dtype和shape才能保证数据一致。
#     并且tofile函数不管数组的排列顺序是C语言格式的还是Fortran语言格式的，统一使用C语言格式输出。
# 2. sep关键字参数:此外如果fromfile和tofile函数调用时指定了sep关键字参数的话，
# 数组将以文本格式输入输出。{这样就可以通过notepad++打开查看, 不过数据是一行显示，不便于查看}
# user_item_mat.tofile(user_item_mat_filename, sep=' ')
    '''
    a =arange(0,12)
    a.shape = 3,4
    print a
    a.tofile("a.bin")
    b = fromfile("a.bin", dtype=float) # 按照float类型读入数据
    print b # 读入的数据是错误的

    print a.dtype # 查看a的dtype
    b = fromfile("a.bin", dtype=int32) # 按照int32类型读入数据
    print b # 数据是一维的
    b.shape = 3, 4 # 按照a的shape修改b的shape
    print b
    '''
# numpy.load和numpy.save函数（推荐在不需要查看保存数据的情况下使用）
# 以NumPy专用的二进制类型保存数据，这两个函数会自动处理元素类型和shape等信息，
# 使用它们读写数组就方便多了，但是numpy.save输出的文件很难和其它语言编写的程序读入：
    '''
    a =arange(0,12)
    a.shape = 3,4
    print a
    save('a.npy',a)
    c=load('a.npy')
    print c
    '''
# Note:
# 1. 文件要保存为.npy文件类型，否则会出错
# 2. 保存为numpy专用二进制格式后，就不能用notepad++打开（乱码）看了，这是相对tofile内建函数不好的一点

# numpy.savez函数
# 如果你想将多个数组保存到一个文件中的话，可以使用numpy.savez函数。savez函数的第一个参数是文件名，其后的参数都是需要保存的数组，也可以使用关键字参数为数组起一个名字，非关键字参数传递的数组会自动起名为arr_0, arr_1, ...。savez函数输出的是一个压缩文件(扩展名为npz)，其中每个文件都是一个save函数保存的npy文件，文件名对应于数组名。load函数自动识别npz文件，并且返回一个类似于字典的对象，可以通过数组名作为关键字获取数组的内容：
# 如果你用解压软件打开result.npz文件的话，会发现其中有三个文件：arr_0.npy， arr_1.npy， sin_array.npy，其中分别保存着数组a, b, c的内容。
    '''
    a = array([[1,2,3],[4,5,6]])
    b = arange(0, 1.0, 0.1)
    c = sin(b)
    savez("result.npz", a, b, sin_array = c)
    r =load("result.npz")
    print r["arr_0"] # 数组a
    print r["arr_1"] # 数组b
    print r["sin_array"] # 数组c
    '''

# numpy.savetxt和numpy.loadtxt（推荐需要查看保存数据时使用）
# Note:savetxt缺省按照'%.18e'格式保存数据， 可以修改保存格式为‘%.8f'(小数点后保留8位的浮点数)， ’%d'(整数)等等
# 总结：
# 载入txt文件：numpy.loadtxt()/numpy.savetxt()
# 智能导入文本/csv文件：numpy.genfromtxt()/numpy.recfromcsv()
# 高速，有效率但numpy特有的二进制格式：numpy.save()/numpy.load()
# 博客地址：http://blog.csdn.net/pipisorry/article/details/39088003
    
###########################################################################
    '''
    #混淆位置。如果是多维数组，则混淆一维的。例如下面的arr.
    print random.permutation(10)
    print random.permutation([1, 4, 9, 12, 15])
    
    arr=arange(9).reshape((3,3))
    print arr
    print random.permutation(arr)
    
    print complex(0,100)#=100j
    
    #norm计算向量的范数，scipy.linalg.norm(a, ord=None)[source]
    #第二个参数可选，无代表求模2范数。
    #from scipy.linalg.misc import norm
    print norm([8,6,10])
    '''
    
    ll=[[1,2,3,4,5,6],[3,4,5,6,7,8]]
    print var(ll[0])
    print var(ll,0)#第二个参数为0，表示按列求方差
    print var(ll,1)#第二个参数为1，表示按行求方差
    
    print mean(ll) #全部元素求均值
    print mean(ll,0)#按列求均值
    print mean(ll,1)#按行求均值
    
    
    
    
    
