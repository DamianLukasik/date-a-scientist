import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sys

#Create your df here:
print('Date-A-Scientist')

### Load in the DataFrame
import pandas as pd
print('Loading Dataset')
#I had to change this line 'df = pd.read_csv("profiles.csv") ' to this line:
df = pd.read_csv(r'C:\Users\Damian\Documents\Python Project\profiles.csv')
#df = pd.read_csv("profiles.csv") 

### Explore the Data
print(df.head(10))
#I understand that rows represent people and columns represent their data. Columns "essay" represent answers on questions. 
print(df.job.head())
print('')
#Rows in column 'job' represent their jobs

#Possible responses on column 'job'
print('Possible responses on column \'job\'')
print(df.job.value_counts())
print('')
#Possible responses on column 'orientation'
print('Possible responses on column \'orientation\'')
print(df.orientation.value_counts())
print('')
#Possible responses on column 'education'
print('Possible responses on column \'education\'')
print(df.education.value_counts())
print('')

### Visualize some of the Data
from matplotlib import pyplot as plt
print('Visualization of the Data')
print('Histogram of Age')
plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()
#I tried this code in my own file and I have seen nice histogram

print('Histogram of Smokes')
plt.hist(df.smokes)
plt.xlabel("Smokes")
plt.ylabel("Frequency")
plt.show()

print('Histogram of Pets')
plt.hist(df.pets)
plt.xlabel("Pets")
plt.ylabel("Frequency")
plt.show()

print('Histogram of Drugs')
plt.hist(df.drugs)
plt.xlabel("Drugs")
plt.ylabel("Frequency")
plt.show()
print('')

### Formulate a Question
print('Possible responses on column \'Zodiac signs\'')
print(df.sign.value_counts())
print('')

### Augment your Data

#import np because in my data exist values NaN
import numpy as np

#df and df This is the same?
#all_data = df <- Yes, it's confusing, I resigned all_data
#Drinks
print('Possible responses on column \'Drinks\'')
print(df.drinks.value_counts())
print('')
drink_mapping = {np.nan: 0, "not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

#Smokes
print('Possible responses on column \'Smokes\'')
print(df.smokes.value_counts())
print('')
smokes_mapping = {np.nan: 0, "no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)

#Drugs
print('Possible responses on column \'Drugs\'')
print(df.drugs.value_counts())
print('')
drugs_mapping = {np.nan: 0, "never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)

#Essay
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
#all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
all_essays = all_essays[essay_cols].apply(' '.join, axis=1)

#df["essay_len"] = all_essays.apply(lambda x: len(x)) - I corrected this row
df["essay_len"] = all_essays.apply(len)
print('10 first rows on column \'Essay len\'')
print(df["essay_len"].head(10))
print('')

#Column with average word length
def average_word_length(words):
  words = words.split()
  div = len(words)
  if(div==0):
    return 0
  return sum(len(word) for word in words) / div

#df["avg_word_length"] = all_essays.apply(lambda x: average_word_length(x) )
df["avg_word_length"] = all_essays.apply(average_word_length)
print('10 first rows on column \'Avg word length\'')
print(df["avg_word_length"].head(10))
print('')
#Column with the frequency of the words "I" or "me" appearing in the essays.

def frequency_words(words):
  words = words.split()
  count_words = 0
  for word in words:
    for find in ['I','me']:
      if find==word:
        count_words+=1
  if len(words)==0:
    return 0
  return (count_words/len(words))

#df["frequency_words"] = all_essays.apply(lambda x: frequency_words(x,['I','me']))
df["frequency_words"] = all_essays.apply(frequency_words)
print('10 first rows on column \'Frequency words\'')
print(df["frequency_words"].head(10))
print('')

### Normalize your Data!
from sklearn import preprocessing

feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]

#Print my data
#print(feature_data.values)
x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

### Use Classification Techniques

#function to divide data
def ShareData(arr,size,normalization=True):
  size_training = round(len(arr)*size)
  data_training = []
  data_test = []
  for i in range(len(arr)):
    if i<size_training:
      data_training.append(arr[i])
    else:
      data_test.append(arr[i])
  if normalization==True:
    min_max_scaler = preprocessing.MinMaxScaler()
    if(isinstance(data_training[0], list)==True):
      data_training = min_max_scaler.fit_transform(data_training)
    if(isinstance(data_test[0], list)==True):
      data_test = min_max_scaler.fit_transform(data_test)
  return data_training, data_test

# arrays of results
results = []

#the data exploration
def data_exploration(x,y,xlabel,ylabel):   
  plt.clf()
  plt.scatter(x.values.reshape(-1, 1),y)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.show()

# 1 question
# training labels for first question
sex_mapping = {np.nan: 0, "m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
print('10 first rows on column \'Sex code\'')
print(df["sex_code"].head(10))
print('')
# training points for first question
education_mapping = {np.nan: 0, "graduated from college/university": 1,"graduated from masters program":2,"working on college/university":3,"working on masters program":4,
"graduated from two-year college":5,"graduated from high school":6,"graduated from ph.d program":7,"graduated from law school":8,"working on two-year college":9,
"dropped out of college/university":10,"working on ph.d program":11,"college/university":12,"graduated from space camp":13,"dropped out of space camp":14,
"graduated from med school":15,"working on space camp":16,"working on law school":17,"two-year college":18,"working on med school":19,"dropped out of two-year college":20,
"dropped out of masters program":21,"masters program":22,"dropped out of ph.d program":23,"dropped out of high school":24,"high school":25,"working on high school":26,
"space camp":27,"ph.d program":28,"law school":29,"dropped out of law school":30,"dropped out of med school":31,"med school":32}
df["education_code"] = df.education.map(education_mapping)
#data exploration
data_exploration(df["sex_code"],df.income,"Sex Code","Income")
data_exploration(df["sex_code"],df["education_code"],"Sex Code","Education Code")
data_exploration(df.income,df["education_code"],"Income","Education Code")
#preparing data
training_labels_first_question, test_labels_first_question = ShareData(df["sex_code"],0.6)
training_points = []
for i in range(len(df["education_code"])):
  training_points.append([df["education_code"][i],df.income[i]])
training_points_first_question, test_points_first_question = ShareData(training_points,0.6)

# 2 question
# training labels for second question
print('10 first rows on column \'Education code\'')
print(df["education_code"].head(10))
print('')
# training points for second question
print('10 first rows on column \'Essay len\'')
print(df["essay_len"].head(10))
print('')
#data exploration
data_exploration(df["education_code"],df['essay_len'],"Education Code","Essay len")
#preparing data
training_labels_second_question, test_labels_second_question = ShareData(df["education_code"],0.6)
training_points = []
for i in range(len(df["essay_len"])):
  training_points.append([df["essay_len"][i]])
training_points_second_question, test_points_second_question = ShareData(training_points,0.6)

# 3 question
# training labels for third question
print('Possible responses on column \'Income\'')
print(df.income.value_counts())
print('')
#data exploration
data_exploration(df.income,df["essay_len"],"Income","Essay len")
data_exploration(df.income,df["avg_word_length"],"Income","Avg word length")
#preparing data
training_labels_third_question, test_labels_third_question = ShareData(df.income,0.6)
# training points for third question
training_points = []
for i in range(len(df["essay_len"])):
  training_points.append([df["essay_len"][i],df["avg_word_length"][i]])
training_points_third_question, test_points_third_question = ShareData(training_points,0.6)

# 4 question
# training labels for four question
print('Possible responses on column \'Age\'')
print(df.age.value_counts())
print('')
#data exploration
data_exploration(df.age,df["frequency_words"],"Age","Frequency words")
#preparing data
training_labels_four_question, test_labels_four_question = ShareData(df.age,0.6)
# training points for four question
training_points = []
for i in range(len(df["frequency_words"])):
  training_points.append([df["frequency_words"][i]])
training_points_four_question, test_points_four_question = ShareData(training_points,0.6)


#time to run the model
import time
start_time = time.time()
stop_time = (time.time() - start_time)

## K-Nearest Neighbors ##
from sklearn.neighbors import KNeighborsClassifier
knn_first_question = []
knn_second_question = []
#1. Can we predict sex with education level and income?
for k in range(1, 40): 
  start_time = time.time() 
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_points_first_question,training_labels_first_question)
  guesses_first_question = classifier.predict(test_points_first_question)
  stop_time = (time.time() - start_time)
  knn_first_question.append([guesses_first_question,classifier,[1,"K-Nearest Neighbors"],test_labels_first_question,stop_time])
results.append(knn_first_question)
#2. Can we predict education level with essay text word counts?
for k in range(1, 40):  
  start_time = time.time() 
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_points_second_question,training_labels_second_question) 
  guesses_second_question = classifier.predict(test_points_second_question)
  stop_time = (time.time() - start_time)
  knn_second_question.append([guesses_second_question,classifier,[2,"K-Nearest Neighbors"],test_labels_second_question,stop_time])
results.append(knn_second_question)

## Support Vector Machines ##
from sklearn.svm import SVC
#1. Can we predict sex with education level and income?
start_time = time.time() 
classifier = SVC(kernel = 'linear')
classifier.fit(training_points_first_question,training_labels_first_question)
guesses_first_question = classifier.predict(test_points_first_question)
stop_time = (time.time() - start_time)
results.append([guesses_first_question,classifier,[1,"Support Vector Machines"],test_labels_first_question,stop_time])
#2. Can we predict education level with essay text word counts?
start_time = time.time() 
classifier = SVC(kernel = 'linear')
classifier.fit(training_points_second_question,training_labels_second_question)
guesses_second_question = classifier.predict(test_points_second_question)
stop_time = (time.time() - start_time)
results.append([guesses_second_question,classifier,[2,"Support Vector Machines"],test_labels_second_question,stop_time])

## Naive Bayes ##
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#1. Can we predict sex with education level and income?
start_time = time.time() 
classifier = MultinomialNB()
classifier.fit(training_points_first_question,training_labels_first_question)
guesses_first_question = classifier.predict(test_points_first_question)
stop_time = (time.time() - start_time)
results.append([guesses_first_question,classifier,[1,"Naive Bayes"],test_labels_first_question,stop_time])
#2. Can we predict education level with essay text word counts?
start_time = time.time() 
classifier = MultinomialNB()
classifier.fit(training_points_second_question,training_labels_second_question)
guesses_second_question = classifier.predict(test_points_second_question)
stop_time = (time.time() - start_time)
results.append([guesses_second_question,classifier,[2,"Naive Bayes"],test_labels_second_question,stop_time])

### Use Regression Techniques

## K-Nearest Neighbors Regression with parameter weights = distance ##
from sklearn.neighbors import KNeighborsRegressor
knnr_first_question = []
knnr_second_question = []
# 3. Predict income with length of essays and average word length?
for k in range(1, 40):  
  start_time = time.time() 
  regressor = KNeighborsRegressor(n_neighbors = k, weights = "distance")
  regressor.fit(training_points_third_question,training_labels_third_question)
  guesses_third_question = regressor.predict(test_points_third_question)
  stop_time = (time.time() - start_time)
  knnr_first_question.append([guesses_third_question,classifier,[3,"K-Nearest Neighbors Regression with parameter weights = distance"],test_labels_third_question,stop_time])
results.append(knnr_first_question)
# 4. Predict age with the frequency of "I" or "me" in essays?
for k in range(1, 40):  
  start_time = time.time() 
  regressor = KNeighborsRegressor(n_neighbors = k, weights = "distance")
  regressor.fit(training_points_four_question,training_labels_four_question)
  guesses_four_question = regressor.predict(test_points_four_question)
  stop_time = (time.time() - start_time)
  knnr_second_question.append([guesses_four_question,classifier,[4,"K-Nearest Neighbors Regression with parameter weights = distance"],test_labels_four_question,stop_time])
results.append(knnr_second_question)

## K-Nearest Neighbors Regression with parameter weights = uniform ##
from sklearn.neighbors import KNeighborsRegressor
knnr_first_question = []
knnr_second_question = []
# 3. Predict income with length of essays and average word length?
for k in range(1, 40):  
  start_time = time.time() 
  regressor = KNeighborsRegressor(n_neighbors = k, weights = "uniform")
  regressor.fit(training_points_third_question,training_labels_third_question)
  guesses_third_question = regressor.predict(test_points_third_question)
  stop_time = (time.time() - start_time)
  knnr_first_question.append([guesses_third_question,classifier,[3,"K-Nearest Neighbors Regression with parameter weights = uniform"],test_labels_third_question,stop_time])
results.append(knnr_first_question)
# 4. Predict age with the frequency of "I" or "me" in essays?
for k in range(1, 40):  
  start_time = time.time() 
  regressor = KNeighborsRegressor(n_neighbors = k, weights = "uniform")
  regressor.fit(training_points_four_question,training_labels_four_question)
  guesses_four_question = regressor.predict(test_points_four_question)
  stop_time = (time.time() - start_time)
  knnr_second_question.append([guesses_four_question,classifier,[4,"K-Nearest Neighbors Regression with parameter weights = uniform"],test_labels_four_question,stop_time])
results.append(knnr_second_question)

# 5. We also learned about K-Nearest Neighbors Regression. Which form of regression works better to answer your question?
#?

### Analyze the Accuracy, Precision and Recall
from sklearn.metrics import classification_report, confusion_matrix  

def Find_validation_accuracy(guesses,test_labels):
  num_correct=0
  for i in range(len(guesses)):
    if test_labels[i] == guesses[i]:
      num_correct += 1
  return (num_correct/len(guesses))

#Find the accuracy, precision, and recall of each model you used, and create graphs showing how they changed.

# K-Nearest Neighbors
# Question 1
#Accuracy
num_correct = Find_validation_accuracy(results[0][5][0],results[0][5][3])
print()
print(""+str(results[0][5][2][1])+" - Question "+str(results[0][5][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[0][5][0]
y_test = results[0][5][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
print("time to run the model: "+str(results[0][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[0][k][0],results[0][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Sex Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()
# Question 2
#Accuracy
num_correct = Find_validation_accuracy(results[1][5][0],results[1][5][3])
print()
print(""+str(results[1][5][2][1])+" - Question "+str(results[1][5][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[1][5][0]
y_test = results[1][5][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
print("time to run the model: "+str(results[1][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):   
  num_correct = Find_validation_accuracy(results[1][k][0],results[1][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Education level Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()

# Support Vector Machines
# Question 1
#Accuracy
num_correct = Find_validation_accuracy(results[2][0],results[2][3])
print()
print(""+str(results[2][2][1])+" - Question "+str(results[2][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[2][0]
y_test = results[2][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[2][4]))
# Question 2
#Accuracy
num_correct = Find_validation_accuracy(results[3][0],results[3][3])
print()
print(""+str(results[3][2][1])+" - Question "+str(results[3][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[3][0]
y_test = results[3][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[3][4]))

# Naive Bayes
# Question 1
#Accuracy
num_correct = Find_validation_accuracy(results[4][0],results[4][3])
print()
print(""+str(results[4][2][1])+" - Question "+str(results[4][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[4][0]
y_test = results[4][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[4][4]))
# Question 2
#Accuracy
num_correct = Find_validation_accuracy(results[5][0],results[5][3])
print()
print(""+str(results[5][2][1])+" - Question "+str(results[5][2][0]))
print("Accuracy: "+str(num_correct))
#Precision
#Recall
y_pred = results[5][0]
y_test = results[5][3]
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[5][4]))

# K-Nearest Neighbors Regression with parameter weights = distance
# Question 3
#Accuracy
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[6][k][0],results[6][k][3])
  arr.append(num_correct)
print()
print(""+str(results[6][5][2][1])+" - Question "+str(results[6][5][2][0]))
print("Accuracy: "+str(np.mean(arr)))
#Precision
#Recall
#y_pred = results[6][5][0]
#y_test = results[6][5][3]
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[6][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[6][k][0],results[6][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Income Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()
# Question 4
#Accuracy
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[7][k][0],results[7][k][3])
  arr.append(num_correct)
print()
print(""+str(results[7][5][2][1])+" - Question "+str(results[7][5][2][0]))
print("Accuracy: "+str(np.mean(arr)))
#Precision
#Recall
#y_pred = results[7][5][0]
#y_test = results[7][5][3]
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[7][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[7][k][0],results[7][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Age Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()

# K-Nearest Neighbors Regression with parameter weights = uniform
# Question 3
#Accuracy
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[8][k][0],results[8][k][3])
  arr.append(num_correct)
print()
print(""+str(results[8][5][2][1])+" - Question "+str(results[8][5][2][0]))
print("Accuracy: "+str(np.mean(arr)))
#Precision
#Recall
#y_pred = results[8][5][0]
#y_test = results[8][5][3]
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[8][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[8][k][0],results[8][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Income Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()
# Question 4
#Accuracy
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[9][k][0],results[9][k][3])
  arr.append(num_correct)
print()
print(""+str(results[9][5][2][1])+" - Question "+str(results[9][5][2][0]))
print("Accuracy: "+str(np.mean(arr)))
#Precision
#Recall
#y_pred = results[9][5][0]
#y_test = results[9][5][3]
#print(confusion_matrix(y_test, y_pred))  
#print(classification_report(y_test, y_pred)) 
print("time to run the model: "+str(results[9][5][4]))
#+ Graphs
arr = []
for k in range(0, 39):  
  num_correct = Find_validation_accuracy(results[9][k][0],results[9][k][3])
  arr.append(num_correct)
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 40), arr, color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=10)
plt.title('Age Classifier Accuracy')  
plt.xlabel('K value')  
plt.ylabel('Validation Aaccuracy')  
plt.show()
