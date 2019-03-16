# -*- coding: utf-8 -*-
"""
@file: tongji.py
@author: tianfeihan
@time: 2019-03-01  20:37:32
@description: 
"""
import numpy  as  np
import csv
import cahocanshu.bianliang as bl
filename="../canshuData/"+bl.project[5]+".csv"
att="a"
qizhong=2
cp=0

data=np.loadtxt(filename,delimiter=",")
if att=="a":
    target=data[:,-1]
else:
    target=data[:,0]


print("一共有"+str(len(target)))
# print("其中一类："+str(cp)+"另一类："+str(len(target)-cp))
print("其中一类："+str(target.sum())+"另一类："+str(len(target)-target.sum()))