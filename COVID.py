import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as py
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import matplotlib
import pickle
from matplotlib import style, pyplot as plt
from sklearn import linear_model


data= pd.read_csv("COVID19_line_list_data.csv", sep=",")
#print(data)


le= preprocessing.LabelEncoder()
reporting_date= le.fit_transform(list(data["reporting_date"]))
country= le.fit_transform(list(data["country"]))
gender= le.fit_transform(list(data["gender"]))
age= le.fit_transform(list(data["age"]))
death= le.fit_transform(list(data["death"]))
recovered= le.fit_transform(list(data["recovered"]))
print(age)
predict= "recovered"   #class label

x=list(zip(reporting_date,country,gender,age,death))

y=list(recovered)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.2)

linear=KNeighborsClassifier(9)
linear.fit(x_train,y_train)
predictions=linear.predict(x_test)
acc=linear.score(x_test,y_test)

print(acc*100)
agew= data[data["country"]=="China"]
print(agew)
agewise=data.groupby(["age"]).agg({"recovered":"sum"})

print(agewise)


plt.figure(figsize=(12,6))
plt.plot(agewise["recovered"],marker='o',label="Recovered cases ")
plt.ylabel("Number of patients")
plt.xlabel("age")
plt.legend()

plt.xticks(rotation=90)
plt.show()




