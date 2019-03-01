# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-13  13:08:30
@description: 数据集缺失时，用每种属相的平均值作为填补
"""

from sklearn.preprocessing import Imputer
import numpy as np
# from ganzangjibing import shujuchuli
# filename="../dataFile/raw_data.csv"
filename="../cahocanshu/hepatitis.csv"

data=np.genfromtxt(filename,skip_header=False,delimiter=',')
# data=data[:,]
print(data)
# data,target=shujuchuli.getdata(filename,"a")
imp=Imputer(missing_values=np.nan,strategy='mean',axis=0)
imp.fit(data)
outerfile=imp.transform(data)
print(outerfile)
np.savetxt("../cahocanshu/hepatitis_after.csv",outerfile,delimiter=',')