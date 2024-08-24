import cv2
import numpy as np
import os

# Load dataset
data_dir = r'C:\Users\DELL\Palmprint_recognition\Dataset'
labels = []
images = []
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    for image_file in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_file)
        if not os.path.isfile(image_path) or not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Skip non-image files
            continue
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
             image = cv2.resize(image, (200, 200))
        else:
            print("Failed to read file:", image_path)
            continue
        images.append(image)
        labels.append(label)
# Convert images and labels lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)
# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Map non-numeric labels to integer values
label_dict = {'Ariana': 0, 'Aurora': 1, 'Biden': 2, 'Bob': 3, 'Joe': 4, 'John': 5, 'Julie': 6, 'Marina': 7, 'Sanna': 8, 'Sofie': 9}

# Convert labels array to integer data type
labels = [label_dict[label] for label in labels]
labels = np.array(labels).astype(int)
images = np.array(images)
print(images.shape)
print(labels.shape)

# Train recognizer
recognizer.train(images, labels)
# Test recognizer
test_image_path = r'C:\Users\DELL\Palmprint_recognition\Dataset\Biden\0006_m_l_06.jpg'
test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
if test_image is not None:
    label, confidence = recognizer.predict(test_image)
    predicted_person = list(label_dict.keys())[list(label_dict.values()).index(label)]

    print("Predicted person: ", predicted_person)
    print("confidence: {:.1f}%".format(confidence))
else:
    print(f"Error reading image at {test_image_path}")