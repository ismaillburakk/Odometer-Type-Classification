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
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

def flatten_images(images):
    return np.array([img.flatten() for img in images])


def cuttingImage(image, xml_tree):
    # Getting info about boundingBox from XML
    bounding_boxes = []
    for obj in xml_tree.findall('.//object'):
        name = obj.find('name').text
        if name != 'odometer' and name != 'X':
            xmin = int(float(obj.find('bndbox/xmin').text))
            ymin = int(float(obj.find('bndbox/ymin').text))
            xmax = int(float(obj.find('bndbox/xmax').text))
            ymax = int(float(obj.find('bndbox/ymax').text))
            bounding_boxes.append((xmin, ymin, xmax, ymax))

    mask = np.zeros_like(image, dtype=np.uint8)
    for xmin, ymin, xmax, ymax in bounding_boxes:
        mask[ymin:ymax, xmin:xmax, :] = 255
    result = cv2.bitwise_and(image, mask)
    return result

image_dir = "trodo-v01/images"
xml_dir = "trodo-v01/pascal voc 1.1/Annotations"

images = os.listdir(image_dir)
xml_files = os.listdir(xml_dir)
xml_files.sort()  # Dosya adlarını sırala

data=[]
j = 1
for i in images:
    try:
        img = cv2.imread(os.path.join(image_dir, i))
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {i}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        xml_file_path = os.path.join(xml_dir, xml_files[j])

        # XML dosyasını açın ve ağacı oluşturun
        tree = ET.parse(xml_file_path)

        # cuttingImage fonksiyonunu çağırın
        final_image = cuttingImage(img, tree)
        j += 1
        data.append(np.array(final_image))
    except Exception as e:
        print(f"Hata: {e}")

#Resize list of data    
resized_data = [cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA) for img in data]    

#Visualization of Data
first_image = resized_data[20]
plt.imshow(first_image)
plt.show()

#flatten
flattened_data = flatten_images(resized_data)

# Labels
json_file_path = "trodo-v01/ground truth/groundtruth.json"
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract mileage values as strings
mileage_strings = [odometer["mileage"] for odometer in data["odometers"]]

# Convert mileage strings to floats (retain decimal information)
labels = [float(mileage_str) for mileage_str in mileage_strings]

#Normalization
normalized_data = [img / 255.0 for img in flattened_data]

#splitting data
x_train, x_test, y_train, y_test = train_test_split(normalized_data, labels, test_size=0.3, random_state=111)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel='linear', C=1.0),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    "Ridge Regressor": Ridge(alpha=1.0)
}

for model_name, model in regression_models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'{model_name} Modelin Mean Absolute Error (MAE) değeri: {mae:.2f}')
