import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PIL import Image
import json
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def flatten_images(images):
    return np.array([img.flatten() for img in images])


def cuttingImage(image, xml_tree):
    # Getting info about boundingBox from XML
    bounding_boxes = []
    for obj in xml_tree.findall('.//object'):
        xmin = int(float(obj.find('bndbox/xmin').text))
        ymin = int(float(obj.find('bndbox/ymin').text))
        xmax = int(float(obj.find('bndbox/xmax').text))
        ymax = int(float(obj.find('bndbox/ymax').text))
        bounding_boxes.append((xmin, ymin, xmax, ymax))

    #painting picture except odometer
    mask = np.zeros_like(image, dtype=np.uint8)
    for xmin, ymin, xmax, ymax in bounding_boxes:
        mask[ymin:ymax, xmin:xmax, :] = 255

    result = cv2.bitwise_and(image, mask)
    return result

image_dir = "trodo-v01/images"
xml_dir = "trodo-v01/pascal voc 1.1/Annotations"

images = os.listdir(image_dir)
xml_files = os.listdir(xml_dir)
xml_files.sort()  

data=[]
j = 1
for i in images:
    try:
        img = cv2.imread(os.path.join(image_dir, i))
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {i}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xml_file_path = os.path.join(xml_dir, xml_files[j])
        tree = ET.parse(xml_file_path)

        # Painting all picture 
        final_image = cuttingImage(img, tree)
        j += 1
        data.append(np.array(final_image))
    except Exception as e:
        print(f"Hata: {e}")

#Resize list of data    
resized_data = [cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA) for img in data]    

#Visualization of Data after painting
first_image = resized_data[4]
plt.imshow(first_image)
plt.show()

#flatten
flattened_data = flatten_images(resized_data)

#Reading labels from ground_truth
json_file_path="trodo-v01/ground truth/groundtruth.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)

labels = [odometer["odometer_type"] for odometer in data["odometers"]]

#LabelEncoding
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

#Normalization
normalized_data = [img / 255.0 for img in flattened_data]

#splitting data
x_train, x_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.3, random_state=111)

models = {
    "SVM": SVC(kernel='linear', C=1.0, random_state=111),
    "Decision Tree": DecisionTreeClassifier(random_state=111),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=111),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=13),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=111)
}

#Fitting Models
for model_name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')
    print(f'{model_name} Precision: {precision * 100:.2f}%')
    print(f'{model_name} Recall: {recall * 100:.2f}%')
    print(f'{model_name} F1 Score: {f1 * 100:.2f}%\n')
    cm = confusion_matrix(y_test, y_pred)
    print(f'{model_name} Confusion Matrix:\n{cm}\n')






