import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("formatteddata.csv")

#split dataset into features and target variable
feature_cols = ['TypeOfAccident ', 'RoadCondition', 'WeatherCondition ', 'speedlimit']
a = df[feature_cols] # Features
b = df.Reason # Target variable
# Split dataset into training set and test set
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=1)
#60%training data and 40%test data


a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.4, random_state=1)

gnb = GaussianNB()
gnb.fit(a_train, b_train)

cnf_mat=[[0]*3]*3
b_pred = gnb.predict(a_test)
accuracy = (feature_cols.accuracy_score(b_test, b_pred)*100)
print("Gaussian Naive Bayes model accuracy(in %):",accuracy)