import sklearn
from sklearn.utils import shuffle
import pandas as pd
import numpy as py
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import matplotlib
import pickle
from matplotlib import style
from matplotlib import pyplot as plt
from sklearn import linear_model

data=pd.read_csv("covid_19_data.csv",sep=",")
#data=data[["ObservationDate","Country_Region","Confirmed","Deaths"]]

le= preprocessing.LabelEncoder()
ObservationDate= le.fit_transform(list(data["ObservationDate"]))
Country_Region = le.fit_transform(list(data["Country_Region"]))
Confirmed = le.fit_transform(list(data["Confirmed"]))
#age= le.fit_transform(list(data["age"]))
#Deaths= le.fit_transform(list(data["Deaths"]))
Recovered= le.fit_transform(list(data["Recovered"]))
#print(Recovered)

predict="Recovered"
a=list(zip(ObservationDate,Country_Region,Confirmed,Recovered))
x=py.array(a)
y=py.array(data["Deaths"])
print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
'''best=0                 #only used for first time running the program after that the best model is pickled 
for hi in range (100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y,test_size=0.1)
#print(x_train,y_test)
    linear=KNeighborsClassifier(9)
    linear.fit(x_train,y_train)
    acc=linear.score(x_test,y_test)
    #print(acc)

    if best<acc:
        best=acc
        with open("COVID.pickle","wb") as f:
            pickle.dump(linear,f)'''

pickle_in=open("COVID.pickle","rb")
linear=pickle.load(pickle_in)
#print("Final accuracy:",best)

india_data=data[data["Country_Region"]=="India"]
datewise_india=india_data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})
print (datewise_india)




plt.figure(figsize=(10,6))
#plt.scatter(data["Country_Region"],data["Recovered"])
plt.plot(datewise_india["Confirmed"],marker='o',label="Confirmed Cases")
plt.plot(datewise_india["Recovered"],marker='*',label="Recovered Cases")
plt.plot(datewise_india["Deaths"],marker='+',label="Death Cases")

plt.ylabel("Number of Patients")
plt.xlabel("Date")
plt.legend()
plt.title("Growth Rate Plot for different Types of cases in India")
plt.xticks(rotation=90)
#plt.show()

prediction = linear.predict(x_test)
#print(prediction)

for y in range(len(prediction)):
    print("predicted:",prediction[y],"data:",x_test[y],"Actual:",y_test[y])

#x_predict=("ObservationDate"=="25","Country_Region"=="114","Confirmed"=="89","Recovered"=="30")

#hello=["ObservationDate"=="03/23/2020","Country_Region"=="India","Confirmed"=="499","Recovered"=="34"]
#x_predict=linear.predict([hello])

x_predict=le.fit_transform([["ObservationDate"=="03/23/2020","Country_Region"=="India","Confirmed"=="499","Recovered"=="34"]])



print(x_predict)