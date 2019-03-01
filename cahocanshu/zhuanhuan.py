# -*- coding: utf-8 -*-
"""
@file: zhuanhuan.py
@author: tianfeihan
@time: 2019-03-01  20:56:32
@description: 将非数值类数据转换为数值型
"""

import pandas as pd
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
if __name__ == '__main__':
    # 定义一个DataFrame数据

    filename="data.csv"
    columns = [ "handicapped", "water", "adoption", "physician", "el", "religious",
                    "anti", "aid", "mx", "class",]

    df = pd.read_csv(filename, names=columns)


    class_label = LabelEncoder()
    columns_w = ["handicapped", "water", "adoption", "physician", "el", "religious",
                    "anti", "aid", "mx" ]
    columns_y = [ "class"]
    bruises_mapping = {"negative": 1, "positive": 0}
    # gill_spacing_mapping = {"w": 1, "c": 0,"d":2}
    # gill_size_mapping={"b": 1, "n": 0}
    # stalk_shape_mapping={"e": 1, "t": 0}
    # ring_number_mapping={"o": 1, "n": 0,"t":2}
    # population_mapping={"a": 6, "c": 5,"n":4,"s": 3, "v": 2,"y":1}
    df["class"]=df["class"].map(bruises_mapping)
    # df["gill-spacing"] = df["gill-spacing"].map(gill_spacing_mapping)
    # df["gill-size"] = df["gill-size"].map(gill_size_mapping)
    # df["stalk-shape"] = df["stalk-shape"].map(stalk_shape_mapping)
    # df["ring-number"] = df["ring-number"].map(ring_number_mapping)
    # df["population"] = df["population"].map(population_mapping)






    for i in columns_w:

        df[i] = class_label.fit_transform(df[i].values)

    df.to_csv('tictac.csv',index=False, header=False)


