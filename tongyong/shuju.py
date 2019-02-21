from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.model_selection import train_test_split

filename="raw_data.csv"
data=np.genfromtxt(filename,skip_header=False,delimiter=',')
x,y=data[:, :-1],data[:, -1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)
print(sum(y_test))
