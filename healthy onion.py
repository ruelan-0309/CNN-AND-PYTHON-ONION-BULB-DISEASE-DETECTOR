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
    "https://imgur.com/5Tn27wM",
    "https://imgur.com/iilqpue",
    "https://imgur.com/Lqxx1iA",
    "https://imgur.com/IUiuP24",
    "https://imgur.com/qAf51Ec",
    "https://imgur.com/QhiLTLA",
    "https://imgur.com/Uf3tteB",
    "https://imgur.com/tYV6BYd",
    "https://imgur.com/61ZoUeP",
    "https://imgur.com/bL2Pa7c",
    "https://imgur.com/76QcetJ",
    "https://imgur.com/cpdcGG6",
    "https://imgur.com/5AuuU2n",
    "https://imgur.com/ZsYRvlM",
    "https://imgur.com/jJ414of",
    "https://imgur.com/AthRCsk",
    "https://imgur.com/Kd3QnxJ",
    "https://imgur.com/m1egTLO",
    "https://i.imgur.com/hN4bMLt.jpeg",
    "https://i.imgur.com/YIIvK8P.jpeg",
    "https://i.imgur.com/Bh9SHqH.jpeg",
    "https://i.imgur.com/FWkRzYN.jpeg",
    "https://i.imgur.com/nbgnDHx.jpeg",
    "https://i.imgur.com/ZBv9u1M.jpeg",
    "https://i.imgur.com/mAAU7qD.jpeg",
    "https://i.imgur.com/pdQL0pb.jpeg",
    "https://i.imgur.com/84shsUP.jpeg",
    "https://i.imgur.com/BqawWil.jpeg",
    "https://i.imgur.com/jiJuz1f.jpeg",
    "https://i.imgur.com/fjjnMW4.jpeg",
    "https://i.imgur.com/fkEPgxV.jpeg",
    "https://i.imgur.com/I5dH3li.jpeg",
    "https://i.imgur.com/wv35g3u.jpeg",
    "https://i.imgur.com/hZtgDcg.jpeg",
    "https://i.imgur.com/Bl1evjM.jpeg",
    "https://i.imgur.com/f7pQNNd.jpeg",
    "https://i.imgur.com/v6rWVzS.jpeg",
    "https://i.imgur.com/sSV7yKl.jpeg",
    "https://i.imgur.com/ol8wJiD.jpeg",
    "https://i.imgur.com/jzJIJ4f.jpeg",
    "https://i.imgur.com/l7ZT8qZ.jpeg",
    "https://i.imgur.com/YTofDFE.jpeg",
    "https://i.imgur.com/8W6QUdD.jpeg",
    "https://i.imgur.com/XVvUDFq.jpeg",
    "https://i.imgur.com/RDmJgx9.jpeg",
    "https://i.imgur.com/ICudlT9.jpeg",
    "https://i.imgur.com/zeo09sK.jpeg",
    "https://i.imgur.com/z2xgwaR.jpeg",
    "https://i.imgur.com/pRN1zHR.jpeg",
    "https://i.imgur.com/rEXVNZ4.jpeg",
    "https://i.imgur.com/5Q4NQoK.jpeg",
    "https://i.imgur.com/v4EdeCZ.jpeg",
    "https://i.imgur.com/UeeUbxk.jpeg",
    "https://i.imgur.com/HkS70PP.jpeg",
    "https://i.imgur.com/fzrIqIR.jpeg",
    "https://i.imgur.com/WXgUxrX.jpeg",
    "https://i.imgur.com/wppU4yc.jpeg",
    "https://i.imgur.com/wMGj61B.jpeg",
    "https://i.imgur.com/FUHieQ2.jpeg",
    "https://i.imgur.com/ms440DP.jpeg",
    "https://i.imgur.com/JK4Yhds.jpeg",
    "https://i.imgur.com/LuVnSnL.jpeg",
    "https://i.imgur.com/REYk0cu.jpeg",
    "https://i.imgur.com/KmXiDie.jpeg",
    "https://i.imgur.com/uddnjWl.jpeg",
    "https://i.imgur.com/JtQ7Du6.jpeg",
    "https://i.imgur.com/LU4pK2g.jpeg",
    "https://i.imgur.com/nyat2B0.jpeg",
    "https://i.imgur.com/zcO1Xc0.jpeg",
    "https://i.imgur.com/rH9CJJ2.jpeg",
    "https://i.imgur.com/yGOKeIk.jpeg",
    "https://i.imgur.com/K7RayCX.jpeg",
    "https://i.imgur.com/rRSywC5.jpeg",
    "https://i.imgur.com/SqSvBHO.jpeg",
    "https://i.imgur.com/0F6BBMv.jpeg",
    "https://i.imgur.com/FGLibXA.jpeg",
    "https://i.imgur.com/BB4KR0Q.jpeg",
    "https://i.imgur.com/82zpQoz.jpeg"
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
