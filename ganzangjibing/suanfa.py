from sklearn.ensemble import AdaBoostClassifier
from general import shujuchuli

#
filename="liver-disorders.csv"
data,target= shujuchuli.getdata(filename, "a")
shuzi=int(len(target)/5*3)

train_data=data[:shuzi,:]
ts_data=data[shuzi:,:]
train_target=target[:shuzi]
ts_target=target[shuzi:]

# 支持向量机 70/345
# clf=svm.SVC()
# clf=clf.fit(train_data,train_target)
# result=clf.predict(ts_data)

#高斯朴素贝叶斯 77/345
# gnb=GaussianNB()
# gnb=gnb.fit(train_data,train_target)
# result=gnb.predict(ts_data)

# 决策树
# clf=tree.DecisionTreeClassifier()
# clf=clf.fit(train_data,train_target)
# result=clf.predict(ts_data)

# adaboost 的集成方法
# clf=AdaBoostClassifier(n_estimators=10)
# scores=cross_val_score(clf,train_data,train_target)
# scores.mean()

# 随机森林的集成方法
clf=AdaBoostClassifier(n_estimators=20)
clf=clf.fit(train_data,train_target)
result=clf.predict(ts_data)

# print(train_data)
# print(train_target)
print(ts_target)
print(result)

print("Number of mislabeled points out of a total %d points : %d" % (len(ts_target), (ts_target != result).sum()),)