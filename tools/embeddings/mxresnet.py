import logging
import time

import cv2
import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon.model_zoo import vision

# Загрузка модели из model_zoo
model = vision.resnet34_v2(pretrained=True)  # Здесь можно использовать другую модель
model.hybridize()

# Конвертация изображения в MXNet NDArray
def preprocess_mxnet(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32) / 127.5 - 1
    return nd.array(img).expand_dims(axis=0)

image_path = "img.png"

# Получение эмбеддингов через MXNet
input_tensor_mx = preprocess_mxnet(image_path)
start = time.time()
for i in range(0, 100):
    embedding_mx = model(input_tensor_mx)
    logging.error(msg="embedding_mx: {}".format(embedding_mx.asnumpy()))
    # print("MXNet Embedding Shape:", embedding_mx.shape)
print("MXNet Time Taken:", time.time() - start)
