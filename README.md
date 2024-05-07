# Odometer Classification and Mileage Extraction using Machine Learning

## Introduction
This report covers two fundamental tasks using the TRODO dataset: Odometer Type 
Classification and Mileage Extraction. In the first task, various machine learning 
algorithms were employed to classify odometers into analog and digital types. The 
second task involved applying different machine learning algorithms to extract mileage 
values from images.

## Task 1: Odometer Type Classification

### Data Exploration and Preprocessing
During data preparation for the odometer type classification task using the TRODO 
dataset, the following steps were undertaken:
1. Reading Images and Labels: Image and label files were read from the specified 
folders, and images were annotated using bounding box information from the 
respective XML files.
2. Filling Non-Bounding Box Areas with Black: Considering that the models to be 
developed should not require information from outside the odometer region, 
areas outside the bounding box were filled with black. This ensures that the 
model focuses solely on information within the odometer region during 
classification.
3. Image Resizing and Normalization: The obtained images were resized to 
predefined dimensions and normalized between 0 and 1.
4. Data Splitting: Odometer types and labels were extracted from the ground truth 
data in the JSON file, and the dataset was divided into training and testing sets.
These steps were adapted for each machine learning model, serving as the input data 
before model training

![image](https://github.com/ismaillburakk/Odometer-Type-Classification/assets/75124682/b8204ba9-7c54-4264-a8c7-516a71e786b4)

### Models
As observed, the performance metrics for odometer type classification using Support 
Vector Machine (SVM), Decision Tree, Random Forest, K-Nearest Neighbors, and 
Logistic Regression are as follows:
1. SVM: Accuracy: 81.31%, Precision: 81.08%, Recall: 81.31%, F1 Score: 80.66%, Confusion Matrix:
[[151 94]
[ 40 432]] 
2. Decision Tree: Accuracy: 80.33%, Precision: 80.15%, Recall: 80.33%, F1 Score: 80.22%, Confusion Matrix:
[[169 76]
[ 65 407]] 
3. Random Forest: Accuracy: 90.66%, Precision: 90.59%, Recall: 90.66%, F1 Score: 90.59%, Confusion Matrix:
[[205 40]
[ 27 445]] 
5. K-Nearest Neighbors: Accuracy: 80.06%, Precision: 81.33%, Recall: 80.06%, F1 Score: 78.29%, Confusion Matrix:
[[120 125] 
[ 18 454]] 
6. Logistic Regression: Accuracy: 83.68%, Precision: 83.47%, Recall: 83.68%, F1 Score: 83.32%, Confusion Matrix:
[[168 77] 
[ 40 432]] 
![image](https://github.com/ismaillburakk/Odometer-Type-Classification/assets/75124682/7c6b4ad8-af45-4740-a0f9-e4c66cbc8df5)

### Evaluation
These results indicate that the Random Forest model achieved the highest accuracy and 
F1 score in odometer type classification. However, examining the confusion matrix for 
each model reveals instances where specific classes were misclassified, warranting a 
more detailed evaluation of the strengths and weaknesses of each model.

## Task 2: Mileage Extraction

### Data Exploration and Preprocessing
During data preparation for the mileage extraction task using the TRODO dataset, the 
following steps were undertaken:
1. Reading Images and Labels: Image and label files were read from the specified 
folders, and images were annotated using bounding box information from the 
respective XML files.
2. Only areas outside the bounding boxes containing numerical values were filled 
with black. This specific approach ensures that only regions with relevant 
numerical information within the bounding boxes retain their color, while the rest 
of the image is blacked out. This approach is designed to focus the model solely 
on the regions containing mileage information during the extraction process.
3. Image Resizing and Normalization: The obtained images were resized to 
predefined dimensions and normalized between 0 and 1.
4. Data Splitting: Mileage values were extracted as strings from the ground truth 
data in the JSON file, and the dataset was divided into training and testing sets.
These steps were adapted for each regression model, serving as the input data before 
model training


![image](https://github.com/ismaillburakk/Odometer-Type-Classification/assets/75124682/b3405be9-596c-4f20-8c79-3051afe82c91)

### Models
As observed, the performance metrics for mileage extraction using Linear Regression, 
Support Vector Regressor, Decision Tree Regressor, K-Nearest Neighbors Regressor, and 
Ridge Regressor are as follows:
1. Linear Regression: Mean Absolute Error (MAE): 121224.54
2. Support Vector Regressor: Mean Absolute Error (MAE): 86221.98
3. Decision Tree Regressor: Mean Absolute Error (MAE): 89807.44
4. K-Nearest Neighbors Regressor: Mean Absolute Error (MAE): 85119.55
5. Ridge Regressor: Mean Absolute Error (MAE): 115109.03

![image](https://github.com/ismaillburakk/Odometer-Type-Classification/assets/75124682/ef5701d1-56ea-4198-a8b8-d0a27b6e92b0)

### Evaluation
The observed MAE values reflect the success of each regression model in the mileage 
extraction task. The Support Vector Regressor has the lowest MAE, indicating better 
predictive ability compared to other models. However, differences in MAE values should 
be considered alongside overall model performance.

### General Evaluation
Upon reviewing the results of both tasks, it is evident that the Random Forest model 
excelled in classification, while the Support Vector Regressor performed well in mileage 
extraction. Nevertheless, there are areas for improvement in the performance of each 
model, which could be addressed through additional data, enhanced feature 
engineering, or the exploration of different algorithms.

