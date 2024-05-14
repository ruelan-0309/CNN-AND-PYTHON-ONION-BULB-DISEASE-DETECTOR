import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import requests
from PIL import Image
from io import BytesIO


def load_and_preprocess_image(url):
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = img.convert('RGB')
img = img.resize((64, 64)) # Resize image to match model input size
img = np.array(img)
img = img / 255.0 # Normalize pixel values to [0, 1]
return img


image_urls = [
"https://imgur.com/vckN8MN",
"https://imgur.com/n8O2ZMS",
"https://imgur.com/wUgSSWt",
"https://imgur.com/Yv7iJSl",
"https://imgur.com/pq9BY6z",
"https://imgur.com/z3yrVAP",
"https://imgur.com/qmW6lA4",
"https://imgur.com/Jxa668A",
"https://imgur.com/LAMP0qV",
"https://imgur.com/w2470mR",
"https://imgur.com/Wn7lEqk",
"https://imgur.com/Buv1tfF",
"https://imgur.com/PVD7CLe",
"https://imgur.com/aj2OSep",
"https://imgur.com/ZFw9hCz",
"https://imgur.com/vAWO8jG",
"https://imgur.com/Yiz2cCt",
"https://imgur.com/frQN8Ha",
"https://imgur.com/rM8cZh5",
"https://imgur.com/LlW69xN",
"https://imgur.com/jemKbwg",
"https://imgur.com/W5xldzD",
"https://imgur.com/0O0wsM5",
"https://imgur.com/SGuigKh",
"https://imgur.com/jdSMkfk",
"https://imgur.com/Vm9tqwl"
]


images = [load_and_preprocess_image(url) for url in image_urls]


model = Sequential([
Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
MaxPooling2D((2, 2)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D((2, 2)),
Flatten(),
Dense(128, activation='relu'),
Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def predict(image):

image = np.expand_dims(image, axis=0)

prediction = model.predict(image)
return prediction


predictions = [predict(img) for img in images]


def display_images_with_predictions(images, predictions):
classes = ["Not Stemphylium Blight", "Stemphylium Blight"] # Assuming 1 corresponds to Stemphylium Blight in your model output
for i, (img, pred) in enumerate(zip(images, predictions)):
plt.subplot(1, len(images), i + 1)
plt.imshow(img)
plt.title(classes[int(np.round(pred[0]))])
plt.axis('off')
plt.show()


display_images_with_predictions(images, predictions)
