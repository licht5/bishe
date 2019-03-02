# -*- coding: utf-8 -*-
"""
@file: zhexiantu.py
@author: tianfeihan
@time: 2019-02-28  21:29:00
@description:  折线图部分
"""
import numpy as np
from pylab import mpl
import pandas as pd
import matplotlib
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import xlrd
import general.globalVariable as gv
def DrwZhexian():
    myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()
    # excel_filename = "../dataFile/excel/" + gv.project_name + ".xls"
    data = xlrd.open_workbook(gv.excel_filename)
    color = ["lightgreen", "yellow", "lightpink", "black", "red"]
    # style_line=[">",".","<","+","-"]
    for i in range(gv.alg_num):
        cmd = exec("asx%s=1" % i)
        table = data.sheet_by_name(gv.alg_name[i])
        ncols = table.nrows

        x = np.arange(1, ncols)
        print(x)
        # eva=[]
        cmd = fig.add_subplot(3, 3, i + 1)
        for k in range(gv.evaluation_num):
            data_ = table.col_values(k, start_rowx=1)
            print(data_)
            # eva.append(data)
            cmd.plot(x, data_, c=color[k], label=gv.alg_name[k])
        plt.title(u'' + gv.alg_name[i] + '', fontproperties=myfont)
        plt.xlabel(u'不平衡率', fontproperties=myfont)

    # plt.legend()
    plt.savefig('../Step1_picture/zhexian/'+gv.project_name+'.png')
    plt.show()
if __name__ == '__main__':
    DrwZhexian()
