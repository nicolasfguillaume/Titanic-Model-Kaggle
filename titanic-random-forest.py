# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 22:55:14 2015

@author: Nicolas
"""

import csv as csv
import pandas as pd
import numpy as np
import pylab as pl
from sklearn.ensemble import RandomForestClassifier             # Import the random forest package
import matplotlib.pyplot as plt

#------------- 1/ Loading and displaying the data----------------------
df = pd.read_csv('train.csv', header = 0)

print df.describe()   #stats about each colum (count,meanmstd,min,max...)

#-------------- 2/ Filtering the data---------------------------------

print df['Age'][0:10]   #get the first 10 rows of the Age column
print df.Age[0:10]      #idem

print df['Age'].mean()      #get the mean value of Age
print df['Age'].median()    #get the median

print df[ ['Sex','Pclass','Age'] ]   #select only these 3 columns

print df[  df['Age'] > 60  ]     #filter by Age > 60  (pass the criteria of df as a where clause into df)

print df[  df['Age'] > 60  ] [ ['Sex','Pclass','Age','Survived'] ]  #select only these 4 columns and filter by Age > 60

print df[  df['Age'].isnull()  ] [ ['Sex','Pclass','Age'] ]  #select only these 3 columns and filter by Age without value

#let's count the number of male in each class:
for i in range(1,4):
    print i, len(   df[  (df['Sex'] == 'male') & (df['Pclass'] == i)  ]   )
    
#let's plot the histogram of the column Age:
df['Age'].hist()
pl.show()

df['Age'].dropna().hist(bins=16,range=(0,80),alpha=0.5)   #the same as above, but dropping the missing values of Age
pl.show()

#-------------- 3/ Cleaning the data---------------------------------
df['Gender'] = 4                                                             #adding a new column and fill with 4

df['Gender'] = df['Sex'].map( lambda x : x[0].upper()   )                    #fill the Gender column with M or F

df['Gender'] = df['Sex'].map( {'female' : 0 , 'male' : 1}  ).astype(int)      #fill the Gender column with  1 or 0

print df.head(5)

#Deal with the missing values of Age (use the median age that was typical in each passenger class):
median_ages = np.zeros((2,3))
print median_ages

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[ (df['Gender'] == i) & (df['Pclass'] == j+1)  ]['Age'].dropna().median()
print median_ages

df['AgeFill'] = df['Age']               #add a new column AgeFill with the values of Age
print df.head(5)

#display these columns for each row without an age value. Returns the first 10 rows
print df[  df['Age'].isnull() ][ ['Gender','Pclass','Age','AgeFill']    ].head(10)

for i in range(0,2):                    #here, fills the missing AgeFill using the median_ages table
   for j in range(0,3):
       df.loc[ df.Age.isnull() & (df.Gender == i) & ( df.Pclass == j+1 ) , 'AgeFill'   ] = median_ages[i,j]

#and verify
print df[  df['Age'].isnull() ][ ['Gender','Pclass','Age','AgeFill']    ].head(10)

#add a column AgeisNull
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

#-------------- 4/ Feature engineering---------------------------------
# df['FamilySize']  = df['SibSp'] + df['Parch']           # not used

#-------------- 5/ Prepare the data for Machine Learning Algo---------------------------------
print df.dtypes[ df.dtypes.map( lambda x : x=='object'   ) ]    #identify Columns that are not numbers
                                                                #because ML algo takes only numbers
df = df.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','Age'], axis = 1)  #then remove these columes from df

print df.head()

train_data = df.values                                # Pandas sends back an numpy array using the .values method

print train_data

#-------------- 6/ Train the Machine Learning Algo on Training Data-------------------------

# Create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

importances = forest.feature_importances_      # evaluate the importance of each feature

#-------------- 7/ Prepare and clean the Test Data----------------------------------------------------

test_data = pd.read_csv('test.csv', header = 0)

test_data['Gender'] = test_data['Sex'].map( {'female' : 0 , 'male' : 1}  ).astype(int)   #fill the Gender column with  1 or 0

test_data['AgeFill'] = test_data['Age']    #add a new column AgeFill with the values of Age

#here, fills the missing AgeFill using the median_ages table
for i in range(0,2):          
   for j in range(0,3):
       test_data.loc[ test_data.Age.isnull() & (test_data.Gender == i) & ( test_data.Pclass == j+1 ) , 'AgeFill'   ] = median_ages[i,j]

test_data['AgeIsNull'] = pd.isnull(test_data.Age).astype(int)        #add a column AgeisNull

PassengerId_values = test_data['PassengerId'].values         #save PassengerId as a numpy array  (using the .values method). Used later to create the prediction file

test_data = test_data.drop(['PassengerId','Name','Sex','Age','Ticket','Cabin','Embarked'], axis = 1)  #then remove these columes from df

print test_data.info()

#Deal with the missing values of Fare: use the median Fare that was typical in each passenger class
print test_data[  test_data['Fare'].isnull()  ]         # filter by Fare without value

median_fares = np.zeros((2,3))

for i in range(0,2):
    for j in range(0,3):
        median_fares[i,j] = test_data[ (test_data['Gender'] == i) & (test_data['Pclass'] == j+1)  ]['Fare'].dropna().median()
print median_fares

#here, fills the missing Fare using the median_fares table
for i in range(0,2):          
   for j in range(0,3):
       test_data.loc[ test_data.Fare.isnull() & (test_data.Gender == i) & ( test_data.Pclass == j+1 ) , 'Fare'   ] = median_fares[i,j]

print test_data.info()

test_data_array = test_data.values  #Pandas sends back an numpy array using the .values method

print test_data_array

#-------------- 7/ Run the Machine Learning Algo on Test Data------------------------------
# Take the same decision trees and run it on the test data
output = forest.predict(test_data_array)

# the output is a column 'survived/not survived'
print output

#-------------- 8/ Save the prediction in a file------------------------------
predictions_file = open("randomforestmodel.csv", "wb")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])

for i in range( len(output)   ):
    p.writerow([PassengerId_values[i], output[i]])

predictions_file.close()   

#---------------------Histogram of Imoortance--------------------------------
print importances

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
indices = list(df)[1::]          #get here the name of features, as a list
y = importances
N = len(y)
x = range(N)
plt.bar(x, y, color="blue", align = "center")
plt.xticks(range(len(indices)), indices)
plt.show()

