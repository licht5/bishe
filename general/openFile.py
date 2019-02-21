# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-17  21:17:46
@description: 根据文件中属性的类型进行打开，如果是数值型，返回的是numpy数组型的data，如果是string型，返回的是list型的data
"""
import numpy as np
import csv
def  openFile(filename,att_type):
    if att_type=="num":
        data = np.genfromtxt(filename, skip_header=False, delimiter=',')
    elif att_type=="str":
        with open(filename) as f:
            reader = csv.reader(f)
            data=list(reader)
    return data,att_type

