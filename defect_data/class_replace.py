# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-13  13:08:30
@description: 有的数据集类别上的数据不是1，0的形式，这个脚本用于将该类数据集替换为本项目所需的标准1-0 二类类别
"""

import numpy as np
filename="../dataFile/Breast_mass.csv"
data=np.loadtxt(filename,delimiter=",",usecols=(1,2,3,4,5,6,7,8,9,10))
for data_ in data:
    if data_[-1]==4:
        data_[-1]=1
    else:
        data_[-1]=0
np.savetxt("../dataFile/breast_cancer.csv",data,delimiter=',')