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
    img = img.resize((64, 64))  # Resize image to match model input size
    img = np.array(img)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img


image_urls = [
    "https://imgur.com/KONqr3H",
    "https://imgur.com/qAZ09lh",
    "https://imgur.com/oPkY2Go",
    "https://imgur.com/AfjAmO6",
    "https://imgur.com/8QBBLIP",
    "https://imgur.com/epLZeTV",
    "https://imgur.com/TGQuezH",
    "https://imgur.com/0ztJMuf",
    "https://imgur.com/3O4k8cS",
    "https://imgur.com/sJ4uUGB",
    "https://imgur.com/js4MZCM",
    "https://imgur.com/byFVK2T",
    "https://imgur.com/zvo2LKc",
    "https://imgur.com/9H3vcxy",
    "https://imgur.com/NXUa4K0",
    "https://imgur.com/kIDjAzf",
    "https://imgur.com/dVTIKwX",
    "https://imgur.com/hoaLtV7",
    "https://imgur.com/8hVABDZ",
    "https://imgur.com/8LJQ4dK",
    "https://imgur.com/fpuP0NP",
    "https://imgur.com/OZybP6N",
    "https://imgur.com/GsYgUgM",
    "https://imgur.com/2SMYVdZ",
    "https://imgur.com/8LmyB5Q",
    "https://imgur.com/bu400Eq",
    "https://imgur.com/m7DpS3y",
    "https://imgur.com/SBRIwAk",
    "https://imgur.com/v5USCGZ",
    "https://imgur.com/v5USCGZ",
    "https://imgur.com/ZH39jCd",
    "https://imgur.com/TcR86T8",
    "https://imgur.com/Ne3FJpf",
    "https://imgur.com/cRncg5d",
    "https://imgur.com/WaKNJli",
    "https://imgur.com/Gjb6hUw",
    "https://imgur.com/0rKBkFp",
    "https://imgur.com/DSIrPQA",
    "https://imgur.com/Yc4dTvT",
    "https://imgur.com/v9PAhiV",
    "https://imgur.com/0fwZCxu",
    "https://imgur.com/dt4mHZx",
    "https://imgur.com/tci6ZD7",
    "https://imgur.com/upUFbPZ",
    "https://imgur.com/vebykUN",
    "https://imgur.com/MEootvm"
]


images = [load_and_preprocess_image(url) for url in image_urls]

# Define CNN model
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
    # Reshape image to match model input shape (batch size of 1)
    image = np.expand_dims(image, axis=0)
    # Make prediction
    prediction = model.predict(image)
    return prediction


predictions = [predict(img) for img in images]


def display_images_with_predictions(images, predictions):
    classes = ["Not Downy Mildew", "Downy Mildew"]  # Assuming 1 corresponds to Downy Mildew in your model output
    for i, (img, pred) in enumerate(zip(images, predictions)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.title(classes[int(np.round(pred[0]))])
        plt.axis('off')
    plt.show()


display_images_with_predictions(images, predictions)
 
