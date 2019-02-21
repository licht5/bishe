# -*- coding: utf-8 -*-
"""
@file: pretreatmnet.py
@author: tianfeihan
@time: 2019-02-18  14:23:48
@description: 对原始数据进行预处理，throw=True，将缺失的数据进行丢弃，否则，将缺失数据进行修复
"""

import csv
def PreTreatmnet(filename,throw,savename):
    if throw:
        data=[]
        with open(filename) as f:
            reader = csv.reader(f)
            for data_ in reader:
                if (("?" in data)):
                    pass
                else:
                    data.append(data_)
        with open(savename, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in data:
                writer.writerow(row)
    else:
        pass
