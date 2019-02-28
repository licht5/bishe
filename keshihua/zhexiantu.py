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
fig=plt.figure()
data=xlrd.open_workbook(gv.excel_filename)
color=["#FF83FA","#CDB79E","#00C5CD","#030303","#CD0000"]
for i in range(gv.alg_num):
    cmd=exec("asx%s=1" % i)
    table=data.sheet_by_name(gv.alg_name[i])
    ncols=table.ncols
    x=np.arange(1,ncols+1)
    # eva=[]
    cmd = fig.add_subplot(3, 3, i+1)
    for k in range(gv.evaluation_num):
        data_=table.col_values(k,start_rowx=1)
        # eva.append(data)
        cmd.plot(x,data_,c=color[k])

myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams['axes.unicode_minus']=False
plt.title(u'数据集'+gv.project_name+'随不平衡率变化时各指标变化图',fontproperties=myfont)
plt.xlabel(u'算法',fontproperties=myfont)
plt.ylabel(u'auc')
plt.show()