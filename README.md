Cuisine Classification of Restaurants
Project Overview

This project develops a machine learning model to classify restaurants based on their primary cuisine. The objective is to predict the cuisine category of a restaurant using its available attributes, particularly the restaurant name.

The system applies text vectorization and supervised classification algorithms to learn patterns that distinguish different cuisine types.

Objective

To build a classification model that:

Preprocesses the restaurant dataset

Handles missing values

Encodes categorical variables

Splits the dataset into training and testing sets

Trains classification models

Evaluates performance using standard classification metrics

Analyzes performance across different cuisines

Dataset Description

The dataset contains restaurant-level information including:

Restaurant Name

Cuisines

Aggregate Rating

City

Votes

Price Range

Delivery Availability

For classification, the primary cuisine (first listed cuisine) is used as the target variable.

Preprocessing Steps

Removed records with missing cuisine values

Filtered out restaurants with zero rating

Extracted the primary cuisine from multiple cuisine entries

Selected the top 15 most frequent cuisines to reduce class imbalance

Reset dataframe index for consistency

Encoded cuisine labels using LabelEncoder

Feature Engineering

The restaurant name was transformed using TF-IDF vectorization. This allows the model to learn textual patterns that indicate cuisine type. For example:

Names containing “Pizza” are likely classified as Pizza

Names containing “Cafe” are likely classified as Cafe

Names containing “Biryani” are likely classified as North Indian

TF-IDF converts text into numerical feature vectors suitable for machine learning algorithms.

Model Training

Two classification algorithms were used:

Logistic Regression

Random Forest Classifier

The dataset was split into training and testing sets using an 80–20 split.

Evaluation Metrics

Model performance was evaluated using:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Classification report (per cuisine category)

These metrics help measure overall performance as well as performance across individual cuisine classes.

Results Summary

Random Forest achieved higher accuracy compared to Logistic Regression.

Overall accuracy was approximately 62–63% for 15 cuisine categories.

Some cuisines such as Pizza, Ice Cream, and North Indian showed high precision and recall due to strong naming patterns.

Cuisines such as Continental and Mughlai showed lower performance due to overlapping naming conventions and fewer distinctive keywords.

Sample Prediction Output

The model also displays restaurant-level predictions:

Restaurant Name | Actual Cuisine | Predicted Cuisine

This demonstrates that the model correctly maps individual restaurants to cuisine categories.

Challenges and Limitations

Class imbalance: Some cuisines have significantly more samples than others.

Overlapping patterns: Some cuisine names share similar textual features.

Limited features: Only restaurant name was used as the primary predictive feature.

Conclusion

The project successfully implements a cuisine classification system using machine learning. The model demonstrates that textual features from restaurant names are strong predictors of cuisine type.

Random Forest provided better generalization performance compared to Logistic Regression. The system satisfies all required steps including preprocessing, training, evaluation, and performance analysis.