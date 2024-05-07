import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

#import data from Excel csv sheet
df = pd.read_csv(r'C:\Users\Ege Evcil\PycharmProjects\University\venv\diabetes.csv')

#show first 5 records of dataset
df.head()

#return the object type, which is dataframe
type(df)

#display the number of entries, the number and names of the column attributes, the data type and
    #digit placings, and the memory space used
df.info()

#identify impossible values and outliers using boxplot
df.boxplot(rot = 0, boxprops = dict(color = 'blue'), return_type = 'axes', figsize = (30, 8))
plt.title("Box Plot of Diabetes Data") #title of plot
plt.suptitle(" ")
plt.xlabel("Attribute") # xaxis label
plt.ylabel("Measurements (cm)") # yaxis label
plt.show()

#summarization
df.describe()

#smooth impossible values by replacing the value with the mean value
df['Glucose'] = df['Glucose'].replace(0, df.Glucose.mean())
df['BloodPressure'] = df['BloodPressure'].replace(0, df.BloodPressure.mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df.SkinThickness.mean())
df['Insulin'] = df['Insulin'].replace(0, df.Insulin.mean())
df['BMI'] = df['BMI'].replace(0, df.BMI.mean())


#summarization and confirmation
df.describe()

#detect duplicated records
var = df[df.duplicated(subset=None, keep=False)]

#visualise pairs plot or scatterplot matrix in relation to diabetes outcome
plt.figure(figsize = (12, 10))
g = sns.pairplot(df, hue= 'Outcome', palette='PuBu')
g = g.map_upper(plt.scatter)
g = g.map_lower(sns.kdeplot)

#display the number of entries, the number and names of the column attributes, the data type and
    #digit placings, and the memory space used
df.info()

# list and count the target class label names and their frequency
count = Counter(df['Outcome'])
count.items()


#count of each target class label
plt.figure(figsize = (5, 5))
sns.countplot(x = 'Outcome',data = df , palette = ['lightsteelblue','steelblue'], hue= 'Outcome')
plt.suptitle("Count of Diabetes Outcome")
plt.show()

#compare linear relationships between attributes using correlation coefficient generated using
    #correlation matrix
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), cmap = 'PuBu', annot = True)
plt.show()

#summarization
df.describe()

#classify and model the data using k-Nearest Neighbour (KNN), Decision Tree (DT), and Naive Bayes (NB)
    #machine learning algorithms
df['Outcome'] = df.Outcome.astype(str)
df['Outcome'] = df.Outcome.astype(object)

#split dataset into attributes and labels
X = df.iloc[:, :-1].values # the attributes
y = df.iloc[:, 8].values # the labels

#choose appropriate range of training set proportions
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

#plot decision tree based on information gain
DT = DecisionTreeClassifier(splitter = 'best', criterion = 'entropy', min_samples_leaf = 2)

#use Gaussian method to support continuous data values
NB = GaussianNB()

#choose recommended optimal number of clusters of sqrt(number of records)
KNN = KNeighborsClassifier(n_neighbors = math.ceil(math.sqrt(768)))

#find best training set proportion for the chosen models
plt.figure()
for s in t:
    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s, random_state = 987)
        DT.fit(X_train, y_train) # consider DT scores
        scores.append(DT.score(X_test, y_test))
        NB.fit(X_train, y_train) # consider NB scores
        scores.append(NB.score(X_test, y_test))
        KNN.fit(X_train, y_train) # consider KNN scores
        scores.append(KNN.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')
plt.xlabel('Training Set Proportion') # x axis label
plt.ylabel('Accuracy') # y axis label

#choose train test splits from original dataset as 80% train data and 20% test data for highest accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=987)

#find optimal k number of clusters
k_range = range(1, 25)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
plt.figure()
plt.xlabel('k') # x axis label
plt.ylabel('Accuracy') # y axis label
plt.scatter(k_range, scores) # scatter plot
plt.xticks([0, 5, 10, 15, 20, 25])

#number of records in training set
len(X_train)

#count each outcome in training set
count = Counter(y_train)
print(count.items())

#using k-Nearest Neighbour (KNN) classifier
#choose 7 as the optimal number of clusters
classifierKNN = KNeighborsClassifier(n_neighbors = 15)
classifierKNN.fit(X_train, y_train)

#using Euclidean distance metric
print(classifierKNN.effective_metric_)#value


#using Naive Bayes (NB) classifier
classifierNB = GaussianNB()
classifierNB.fit(X_train, y_train)

#show prior probability of each class
print(classifierNB.class_prior_)


#using Decision Tree (DT) classifier
classifierDT = DecisionTreeClassifier(splitter = 'best', criterion='entropy', min_samples_leaf = 2)
classifierDT.fit(X_train, y_train)


fig = plt.figure(figsize = (55, 20))
fn = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
DT = tree.plot_tree(classifierDT,
                    feature_names = fn,
                    class_names = y,
                    filled = True)

#identifies the important features
print(classifierDT.feature_importances_)#value

#number of records in test set
len(X_test)

#count each outcome in test set
count = Counter(y_test)
print(count.items())

#use the chosen three models to make predictions on test data
y_predKNN = classifierKNN.predict(X_test)
y_predDT = classifierDT.predict(X_test)
y_predNB = classifierNB.predict(X_test)

#for k-Nearest Neighbours model
#using confusion matrix
print(confusion_matrix(y_test, y_predKNN))
print(classification_report(y_test, y_predKNN))

#using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierKNN.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predKNN))

#for Naive Bayes model
#using confusion matrix
print(confusion_matrix(y_test, y_predNB))
print(classification_report(y_test, y_predNB))

#using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierNB.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predNB))

#for Decision Tree model
#using confusion matrix
print(confusion_matrix(y_test, y_predDT))
print(classification_report(y_test, y_predDT))

#using accuracy performance metric
print("Train Accuracy: ", accuracy_score(y_train, classifierDT.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_predDT))

#data to plot
n_groups = 3
algorithms = ('k-Nearest Neighbour (KNN)', 'Decision Tree (DT)', 'Naive Bayes (NB)')
train_accuracy = (accuracy_score(y_train, classifierKNN.predict(X_train))*100,
                  accuracy_score(y_train, classifierDT.predict(X_train))*100,
                  accuracy_score(y_train, classifierNB.predict(X_train))*100)
test_accuracy = (accuracy_score(y_test, y_predKNN)*100,
                 accuracy_score(y_test, y_predDT)*100,
                 accuracy_score(y_test, y_predNB)*100)

#create plot
ax = plt.subplots(figsize=(15, 5))
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8
rects1 = plt.bar(index, train_accuracy, bar_width, alpha = opacity, color='Cornflowerblue', label='Train')
rects2 = plt.bar(index + bar_width, test_accuracy, bar_width, alpha = opacity, color='Teal', label='Test')
plt.xlabel('Algorithm') # x axis label
plt.ylabel('Accuracy (%)') # y axis label
plt.ylim(0, 115)
plt.title('Comparison of Algorithm Accuracies') # plot title
plt.xticks(index + bar_width * 0.5, algorithms) # x axis data labels
plt.legend(loc = 'upper right') # show legend
for index, data in enumerate(train_accuracy):
    plt.text(x = index - 0.035, y = data + 1, s =  format(round(data, 2)), fontdict = dict(fontsize = 8))
for index, data in enumerate(test_accuracy):
    plt.text(x = index + 0.25, y = data + 1, s = format(round(data, 2)), fontdict = dict(fontsize = 8))
plt.show()

#summarization
df.describe()

#new data
newdata = [[1, 50, 80, 33, 70, 30, 0.55, 20]]

#compute probabilities of assigning to each of the two classes of outcome
probaNB = classifierNB.predict_proba(newdata)
probaNB.round(4) # round probabilities to four decimal places, if applicable

#make prediction of class label
predNB = classifierNB.predict(newdata)
