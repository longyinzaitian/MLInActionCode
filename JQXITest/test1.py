#coding=utf-8
'''
Created on 2015年12月17日

@author: 15361
'''
from numpy import *
rowlist='1,2,3\
\
4,5,6'
recodlist=[map(eval,row.split(',')) for row in rowlist.splitlines()]
map1 = map(eval,'00')
print map1
print shape(mat(recodlist))
print recodlist
k=[[1,3],[2,4]]
k=mat(k)
print k
# print k.transpose()
print k[:,0]