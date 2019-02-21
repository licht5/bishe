import pandas as pandas
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
vectorize=CountVectorizer()
train_data=[['w','t','y'],['y','t','y'],['w','t','n'],['w','s','n']]
# train_t=['y','y','n','n']
# ts_data=[['w','s','y'],['y','t','n']]
#
# X=vectorize.fit_transform(train_data)
# print(BaseException.toarray)
c = ['A','A','A','B','B','C','C','C','C']
category = pandas.Categorical(train_data)
#接下来查看category的label即可
print(category)
# print category.labels
#
# gnb=SVC()
# clf=gnb.fit(train_data,train_t)
# result=gnb.predict(ts_data)
# print(result)
