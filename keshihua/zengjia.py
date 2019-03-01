# -*- coding: utf-8 -*-
"""
@file: zengjia.py
@author: tianfeihan
@time: 2019-03-01  16:58:30
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

fig=plt.figure()
data=xlrd.open_workbook(gv.excel_filename)
color=["lightgreen","yellow","lightpink","black","red"]
# style_line=[">",".","<","+","-"]
table=data.sheet_by_name(gv.alg_name[0])
ncols=table.nrows
x=np.arange(1,ncols)
for k in range(gv.evaluation_num):
    data_ = table.col_values(k, start_rowx=1)
    print(data_)
    plt.plot(x, data_, c=color[k], label=gv.alg_name[k])
plt.title(u'' + gv.alg_name[0] + '', fontproperties=myfont)
plt.xlabel(u'不平衡率', fontproperties=myfont)


plt.legend()



plt.show()