from sklearn.ensemble import AdaBoostClassifier
from general import shujuchuli
from sklearn.metrics import classification_report
from sklearn import metrics
# 如果将缺失的数据全部删除掉后，得到的数据是data_after.csv
# 将缺失的数据属性进行填充后得到的数据是ontfile.csv，进行数据处理的是 tongji/queshi.py模块


train_filename="../general/train.csv"
test_filename="../general/test.csv"

train_data,train_target= shujuchuli.getdata(train_filename, "a","num")


ts_data,ts_target= shujuchuli.getdata(test_filename, "a","num")
# shuzi=int(len(target)/5*3)
#
# train_data=data[:shuzi,:]
# ts_data=data[shuzi:,:]
# train_target=target[:shuzi]
# ts_target=target[shuzi:]

# np.savetxt("ts_data.csv",ts_data,delimiter=',')

shuju_1=sum(train_target)
print(shuju_1)
print(len(train_target)-shuju_1)

shuju_2=sum(ts_target)
print(shuju_2)
print(len(ts_target)-shuju_2)

# 支持向量机 70/345
# clf=svm.SVC(probability=True)
# clf=clf.fit(train_data,train_target)
# result=clf.predict(ts_data)

#高斯朴素贝叶斯 77/345
# gnb=GaussianNB()
# clf=gnb.fit(train_data,train_target)
# result=gnb.predict(ts_data)

# 决策树
# clf=tree.DecisionTreeClassifier()
# clf=clf.fit(train_data,train_target)
# result=clf.predict(ts_data)

# adaboost 的集成方法
# clf=AdaBoostClassifier(n_estimators=1)
# clf=clf.fit(train_data,train_target)
# result=clf.predict(ts_data)

# 随机森林的集成方法  n_estimators为5时，准确率最高
clf=AdaBoostClassifier(n_estimators=1)
clf=clf.fit(train_data,train_target)
result=clf.predict(ts_data)

print("acc",metrics.accuracy_score(ts_target,result))
print("pre",metrics.precision_score(ts_target,result))
print("rec",metrics.recall_score(ts_target,result))

# print(train_data)
# print(train_target)
# print(ts_target)
# print(result)

target_names =['class 0', 'class 1']
print(classification_report(ts_target, result, target_names=target_names))
print("f1:",metrics.f1_score(ts_target,result))
print("auc:",metrics.roc_auc_score(ts_target,result))


# predict_prob_y = clf.predict_proba(ts_data)
# new_result=[]
# for i  in predict_prob_y:
#     new_result.append(i[1])
# print("auc:",metrics.roc_auc_score(ts_target,np.array(new_result)))
