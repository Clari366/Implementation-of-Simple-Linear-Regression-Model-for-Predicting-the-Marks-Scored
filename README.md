# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import the needed packages.
2.Assigning hours to x and scores to y.
3.Plot the scatter plot.
4.Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: CLARISSA K
RegisterNumber: 212224230047 
*/

# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:

<img width="157" height="126" alt="319848276-5968514c-05e2-4d71-b132-b0acea5efd47" src="https://github.com/user-attachments/assets/d13c984e-0441-4124-b6b7-0eb564c39fb9" />

<img width="192" height="130" alt="319848309-ea9cb89a-f4b8-473d-84b8-1f92b2ee64a2" src="https://github.com/user-attachments/assets/9b1462ce-4b4b-4c0d-9993-cf0cf9e4fdf5" />


<img width="631" height="487" alt="319848327-9d7409fe-cb21-4727-9bd1-365c85ab9f1a" src="https://github.com/user-attachments/assets/44f71e2b-821c-429d-a450-9a7a0831c799" />


<img width="756" height="72" alt="319848353-44b39961-15f3-4ef1-b64f-01bd7c4fff12" src="https://github.com/user-attachments/assets/d022b905-735b-4261-91e7-6c17a888b4a4" />

<img width="793" height="566" alt="319852122-be5ed2ff-790c-4d0d-84c0-a230f9f4d2df" src="https://github.com/user-attachments/assets/09dae919-833a-47a8-9770-e96e072e1c05" />

<img width="803" height="576" alt="319852143-3871af46-764a-496a-ad7d-39a81d931dee" src="https://github.com/user-attachments/assets/e3710ef4-3a3c-4659-b23a-334f6ffb9fb4" />

<img width="450" height="71" alt="319852157-b2a0e793-2aef-4ae5-ad28-79ec7f2399be" src="https://github.com/user-attachments/assets/d8afc88c-aee4-49e9-98a3-699544e1e2f5" />

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
