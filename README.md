# Road-Accident-Analysis

## Introduction

The cost of injuries has a very great effect on society due to road accidents. Accidents result in thousands of lives lost, hundreds of thousands of injuries and billions of property damage. The primary objective of this project is to know the factors/reasons which significantly affect the severity of driver injuries caused by road accidents, taking the safety measures to avoid accidents and reduce the property damage. The ability to accurately predict the type of motor vehicle crashes with input variables, such as time, weather, location, road type could significantly give us the solution to reduce the severity. 

## Dataset
For this project, we used the data from the National Automotive Sampling System (NASS) General Estimates System (GES). The GES datasets are intended to be a nationally representative probability sample from the estimated accident reports in the year 2017. The total set includes labels of month,  Nature of accident, classification of Accident, Causes, Road feature, Fatal, speed limit, time of the accident. 

To make sure that our data preparation is valid, we have checked the correctness of attribute selection. There are several attribute selection techniques to find a minimum set of attributes so that the resulting probability distribution of the data classes is as close as possible to the original distribution of all attributes.

## Machine Learning Techniques
### Feature Engineering :
* We have the formatted dataset by giving certain codes for the types of attributes such that we can easily identify and classify the data.

* Here, we used the KNN classifier algorithm we splitting the data into training and testing datasets and predict the reason.

* Linear regression for regression tasks.
  * Linear Regression: The objective of a linear regression model is to find a relationship between one or more features(independent variables) and a continuous target variable(dependent variable)

* To determine the best fit/regression line we need to find the line for which the deviation between the predicted values and the observed values is minimum. So we use mean and standard deviation to find this.

Refer : linear_regression.py


### K-Nearest Neighbour classification: 
* The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.
* KNN captures the idea of similarity (sometimes called distance, proximity, or closeness) with some mathematics we might have learned in our childhood— calculating the distance between points on a graph.
* The KNN Algorithm
  * Load the data
  * Initialize K to your chosen number of neighbors
  * For each example in the data
  * Calculate the distance between the query example and the current example from the data.
  * Add the distance and the index of the example to an ordered collection
  * Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
  * Pick the first K entries from the sorted collection
  * Get the labels of the selected K entries
  * If regression, return the mean of the K labels 
  * If classification, return the mode of the K labels

Refer : KNN.py

### Decision Tree:
Refer : decision_tree.py

### Naive Bayes:
Refer : naive_bayes.py

## Observation
By applying three classifiers we get to know that naive bayes classifier is best amongst the three because it is more accurate.


## Conclusion
In this project, we analysed the  National Automotive Sampling System (NASS) General Estimates System (GES) automobile accident dataset and used effective classifier Random Forest to predict reasons for these accidents. The classification accuracy and observations obtained in our experiments reveal that Speed limit plays a very vital role in these accidents. And even weather conditions had a great impact. More information can be identified if more features are available that are closely associated with an accident. Later, prevention methods can also be developed in the locations which are more prone to accidents found from the analysis.
