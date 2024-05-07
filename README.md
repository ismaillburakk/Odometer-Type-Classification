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
