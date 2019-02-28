# -*- coding: utf-8 -*-
"""
@author: tianfeihan
@time: 2019-02-13  12:20:47
@description: 构造一个通用函数SetDataUnbalanced(filename,rate_x,rate_y),此函数用于将格式为csv的file文件划分为不平衡比例为rate，
函数结果能够输出两个文件，一个为测试集，一个训练集
"""
from sklearn.preprocessing import Imputer
import numpy as np
import csv
import random
from sklearn.model_selection import train_test_split

def SetDataUnbalanced(filename,test_rate,rate_x,rate_y,att_type,att_add):

    if att_type=="num":
        data = np.genfromtxt(filename, skip_header=False, delimiter=',')
        if att_add=="a":
            Attributes, targets = data[:, :-1], data[:, -1]
        else:
            Attributes, targets = data[:, 1:], data[:, 0]

    elif att_type=="str":
        with open(filename) as f:
            reader = csv.reader(f)
            data=list(reader)
        Attributes,targets=[],[]
        for i in data:
            if att_add=="a":
                Attributes.append(i[1:])
                targets.append(i[0])
            else:
                Attributes.append(i[:-1])
                targets.append(i[-1])
    else:
        print("error：没有输入合适的属性type！")
        return

    data_len=len(targets)
    test_num=int(test_rate*data_len)
    train_num=data_len-test_num
    positive=targets[0]

    positive_num=int(train_num*rate_x/(rate_x+rate_y))
    negtive_num=train_num-positive_num

    flag_positive=0
    flag_negtive = 0
    train_data=[]
    test_data=[]
    # random_num=random.randint(0,data_len-1)
    for i in range(data_len):
        random_num = random.randint(0, len(data)-1)
        data_tem=data[random_num]
        # del data[random_num]
        np.delete(data,random_num, axis=0)
        if att_add=="a":
            att=data_tem[-1]
        else:
            att=data_tem[0]
        if att == positive and flag_positive < positive_num:
            train_data.append(data_tem)
            flag_positive = flag_positive + 1
        elif att!=positive and flag_negtive<negtive_num:
            train_data.append(data_tem)
            flag_negtive=flag_negtive+1
        else:
            # break
            test_data.append(data_tem)





    #
    # for data_ in data:
    #     if att_add=="a":
    #         att=data_[-1]
    #     else:
    #         att=data_[0]
    #     if att==positive and flag_positive<positive_num:
    #         train_data.append(data_)
    #         flag_positive=flag_positive+1
    #     elif att!=positive and flag_negtive<negtive_num:
    #         train_data.append(data_)
    #         flag_negtive=flag_negtive+1
    #     else:
    #         test_data.append(data_)
    if att_type=="num":
        np.savetxt("../dataFile/train.csv", train_data, delimiter=',')
        np.savetxt("../dataFile/test.csv", test_data, delimiter=',')
    elif att_type=="str":
        with open('../dataFile/train.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in train_data:
                writer.writerow(row)
        with open('../dataFile/test.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for row in test_data:
                writer.writerow(row)

    # print(test_data)






if __name__ == '__main__':
    rate_x, rate_y=1,1
    test_rate=0.3
    # filename="data_after.csv"
    filename="../dataFile/mushroom.csv"

    SetDataUnbalanced(filename,test_rate,rate_x,rate_y,"num","a")