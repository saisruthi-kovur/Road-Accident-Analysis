import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
col_names = ['TypeOfAccident ', 'RoadCondition', 'WeatherCondition ', 'speedlimit']
# load dataset
roadAcc = pd.read_csv("knn-classifier.csv", header=None, names=col_names)

roadAcc.head()

#split dataset in features and target variable
feature_cols = ['Result']
X = roadAcc[feature_cols] # Features
y = roadAcc.label # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
