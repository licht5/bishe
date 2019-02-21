from sklearn.preprocessing import Imputer
import numpy as np
# from ganzangjibing import shujuchuli
filename="raw_data.csv"
data=np.genfromtxt(filename,skip_header=False,delimiter=',')
# data=data[:,]
print(data)
# data,target=shujuchuli.getdata(filename,"a")
imp=Imputer(missing_values=np.nan,strategy='mean',axis=0)
imp.fit(data)
outerfile=imp.transform(data)
print(outerfile)
np.savetxt("ontfile.csv",outerfile,delimiter=',')