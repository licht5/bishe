# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-17  20:37:33
@description:  统计两个类别，共8124个，知悉p类的蘑菇有3916个，e类4208个
"""
import numpy  as  np
import csv
filename="mushroom.csv"
cp=0
with open(filename) as f:
    reader = csv.reader(f)
    for i in reader:
        if i[0]=="p":
            cp=cp+1


print(cp)
print(8124-cp)
