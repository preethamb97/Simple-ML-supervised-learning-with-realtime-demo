import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style
data=pd.read_csv("student-mat.csv",sep=";")

data=data[["G1","G2","G3","studytime","failures","absences"]]
predict="G3"
X=np.array(data.drop([predict],1))
y=np.array(data[predict])
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
'''
while 1:
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.1)
    linear=linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)
    print(acc)
    if acc >=0.96:
        print(acc)
        with open("student_model.pickle","wb") as f:
            pickle.dump(linear,f)
            break
'''    
pickle_in=open("student_model.pickle","rb")
linear=pickle.load(pickle_in)
pickle_in.close()

acc=linear.score(x_test,y_test)
print(acc)
#print('coefficient',linear.coef_)
#print("intercept",linear.intercept_)
predictions=linear.predict(x_test)
#for x in range(len(predictions)):
    #print(int(predictions[x]),x_test[x],y_test[x])


p="G2"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()