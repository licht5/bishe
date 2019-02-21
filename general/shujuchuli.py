# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-13  13:08:30
@description:
"""


import csv
import numpy


# 定义一个将csv数据转换为data与target的函数，file是csv文件名，mark为"b"表示target在每行首，mark为"a"表示target在每行末
def getdata(filename,mark,att_type):
    if att_type=="num":
        my_matrix = numpy.genfromtxt(filename, skip_header=False, delimiter=',')
        if (mark == "a"):
            data = my_matrix[:, :-1]
            target = my_matrix[:, -1]
        elif (mark == "b"):
            data = my_matrix[:, 1:]
            target = my_matrix[:, 0]
    elif att_type=="str":
        with open(filename) as f:
            reader = csv.reader(f)
            my_matrix=list(reader)
        data,target=[],[]
        if (mark == "a"):
            for i in my_matrix:
                data.append(i[0:-1])
                target.append(i[-1])
        elif (mark == "b"):
            for i in my_matrix:
                data.append(i[1:])
                target.append(i[0])
    else:
        print("error：没有输入合适的属性type！")
        return
    return data,target

def getTotalData(filename,train_num,ts_num):
    my_matrix1 = numpy.loadtxt(filename, delimiter=",", skiprows=0)
    length = len(my_matrix1)
    train_nu=int(train_num*length/(train_num+ts_num))
    train_matrix=my_matrix1[:train_nu]
    ts_matrix=my_matrix1[train_nu:]
    return train_matrix,ts_matrix





if __name__ == '__main__':
    filename="data_after.csv"
    ganzdata,ganztarget=getTotalData(filename,3,5)
    print(len(ganzdata),len(ganztarget))
