#coding:utf-8
"""
@file: xiangxing_auc.py
@author: tianfeihan
@time: 2019-02-28  19:19:14
@description: 
"""

import numpy as np
from pylab import mpl
import pandas as pd
import matplotlib
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import xlrd
import general.globalVariable as gv
myfont = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams['axes.unicode_minus']=False

data=xlrd.open_workbook(gv.excel_filename)
to_data={}
f1_data={}
fig=plt.figure()


for i in range(gv.alg_num):
    table=data.sheet_by_name(gv.alg_name[i])
    ncols=table.ncols
    f1_data_=table.col_values(ncols-2,start_rowx=1)
    data_=table.col_values(ncols-1,start_rowx=1)
    f1_data[gv.alg_name[i]]=f1_data_
    to_data[gv.alg_name[i]]=data_
fig.add_subplot(2, 1, 1)
dp=pd.DataFrame(to_data)
dp.boxplot()
plt.title(u'不同算法在数据集'+gv.project_name+'上F1表现箱形图',fontproperties=myfont)
plt.xlabel(u'算法',fontproperties=myfont)
plt.ylabel(u'F1')

fig.add_subplot(2, 1, 2)
cmd2=pd.DataFrame(f1_data)
cmd2.boxplot()

plt.title(u'不同算法在数据集'+gv.project_name+'上auc表现箱形图',fontproperties=myfont)
# plt.yaxis.grid(True)
plt.xlabel(u'算法',fontproperties=myfont)
plt.ylabel(u'auc')
plt.show()