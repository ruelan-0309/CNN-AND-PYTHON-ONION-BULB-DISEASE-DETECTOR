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
"https://imgur.com/JPJfd5n",
"https://imgur.com/Lqo0rWl",
"https://imgur.com/Vvt9PTc",
"https://imgur.com/OEeNpoC",
"https://imgur.com/Iz9wtzt",
"https://imgur.com/6n7Ujr4",
"https://imgur.com/m7rURRA",
"https://imgur.com/PsaJdRm",
"https://imgur.com/UkQUikb",
"https://imgur.com/3kfKRIj",
"https://imgur.com/AmgZbNi",
"https://imgur.com/kcjJPHs",
"https://imgur.com/kcjJPHs",
"https://imgur.com/Ojdsu07",
"https://imgur.com/CTTpvxa",
"https://imgur.com/YJKOeRw",
"https://imgur.com/goRE6Y8",
"https://imgur.com/5MDAKe4",
"https://imgur.com/Bjt0SZB",
"https://imgur.com/r5APyd7",
"https://imgur.com/efc0y7T",
"https://imgur.com/yjrBTPJ",
"https://imgur.com/undefined",
"https://imgur.com/IPXYHsr",
"https://imgur.com/8mXv4C9",
"https://imgur.com/MsCUcrc",
"https://imgur.com/isynT1Y",
"https://imgur.com/p9QrQfQ"
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
classes = ["Not Iris Yellow Spot", "Iris Yellow Spot"] # Assuming 1 corresponds to Iris Yellow Spot in your model output
for i, (img, pred) in enumerate(zip(images, predictions)):
plt.subplot(1, len(images), i + 1)
plt.imshow(img)
plt.title(classes[int(np.round(pred[0]))])
plt.axis('off')
plt.show()


display_images_with_predictions(images, predictions)

