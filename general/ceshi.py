# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-13  13:08:30
@description: 
"""

from sklearn.preprocessing import Imputer
import numpy as np
# from ganzangjibing import shujuchuli
filename="test.csv"
data=np.genfromtxt(filename,skip_header=False,delimiter=',')
x,y=data[:, :-1],data[:, -1]
print(sum(y))
print(len(data))
print(data)
