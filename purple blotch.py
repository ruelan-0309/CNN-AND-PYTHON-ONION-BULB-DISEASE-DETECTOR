import numpy as np
import cv2
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions


def load_image_from_url(url):
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = np.array(img)
return img


def preprocess_image(img):
img = cv2.resize(img, (224, 224))
img = preprocess_input(img)
return img


model = MobileNetV2(weights='imagenet')


image_urls = [
"https://imgur.com/bpb4UQX",
"https://imgur.com/8nvUqdP",
"https://imgur.com/pWOlTuP",
"https://imgur.com/bmHc7mB",
"https://imgur.com/pa0wjnA",
"https://imgur.com/yYVwy0n",
"https://imgur.com/lNgnCGk",
"https://imgur.com/IvVAQZa",
"https://imgur.com/cbtrBxV",
"https://imgur.com/MMlpNS0",
"https://imgur.com/M3ytuks",
"https://imgur.com/Q3ct7hF",
"https://imgur.com/ITw45hi",
"https://imgur.com/WgWKSve",
"https://imgur.com/yl05X54",
"https://imgur.com/Dh351HB",
"https://imgur.com/s0W1FHk",
"https://imgur.com/Sh67Ht6",
"https://imgur.com/3rrqQgX",
"https://imgur.com/DgxArUF",
"https://imgur.com/LnZGvQp",
"https://imgur.com/cazesQ9",
"https://imgur.com/yHcOE2w",
"https://imgur.com/8GZo3TQ",
"https://imgur.com/TKGgokP",
"https://imgur.com/mcz07ph",
"https://imgur.com/lTFCzbQ",
"https://imgur.com/YZXK9Pn",
"https://imgur.com/lb56wcx",
"https://imgur.com/oEjcpmx",
"https://imgur.com/GTESDjt",
"https://imgur.com/oZzEnCR",
"https://imgur.com/gmbRjpS",
"https://imgur.com/vPQXmP7",
"https://imgur.com/6SGciMw",
"https://imgur.com/JTybhrL"
]


for i, url in enumerate(image_urls):

img = load_image_from_url(url)

img = preprocess_image(img)

preds = model.predict(np.expand_dims(img, axis=0))
label = decode_predictions(preds, top=1)[0][0][1]

plt.imshow(img)
plt.title(f"Image {i+1}: {label}")
plt.axis('off')
plt.show()
