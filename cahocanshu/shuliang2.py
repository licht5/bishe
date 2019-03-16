# -*- coding: utf-8 -*-
"""
@file: shuliang2.py
@author: tianfeihan
@time: 2019-03-13  19:55:49
@description: 
"""
from sklearn.model_selection import learning_curve   #可视化学习的整个过程
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits=load_digits()
X=digits.data
y=digits.target

#交叉验证测试
train_sizes,train_loss,test_loss = learning_curve(SVC(gamma=0.1),X,y,cv=10,scoring='neg_mean_squared_error',train_sizes=[0.1,0.25,0.5,0.75,1])   #记录的点是学习过程中的10%，25%等等的点
train_loss_mean = -1 * np.mean(train_loss,axis=1)
test_loss_mean = -1 * np.mean(test_loss,axis=1)

#可视化展示
plt.subplot(1,2,1)
plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='train')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='cross_validation')

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")

#交叉验证测试
train_sizes,train_loss,test_loss = learning_curve(SVC(gamma=0.001),X,y,cv=10,scoring='neg_mean_squared_error',train_sizes=[0.1,0.25,0.5,0.75,1])   #记录的点是学习过程中的10%，25%等等的点
train_loss_mean = -1 * np.mean(train_loss,axis=1)
test_loss_mean = -1 * np.mean(test_loss,axis=1)

#可视化展示
plt.subplot(1,2,2)
plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='train')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='cross_validation')

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
