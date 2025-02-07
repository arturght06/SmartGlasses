import time
import cv2
import numpy as np
import onnxruntime
from numpy.linalg import norm
print(onnxruntime.get_device())  # Should return 'GPU'


# Загрузите модель
session = onnxruntime.InferenceSession("mobilenet/mobilenetv3_large_100_Opset16.onnx", providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"]) # "TensorrtExecutionProvider", 'CPUExecutionProvider'
# session.set_providers(["CUDAExecutionProvider"])

print("Доступные Execution Providers:", onnxruntime.get_available_providers())
print("Используемый Execution Provider:", session.get_providers())


# Посмотрите выходные имена и формы
outputs = session.get_outputs()
for output in outputs:
    print("Имя:", output.name)
    print("Тип:", output.type)
    print("Форма:", output.shape)


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


def prepare_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    img = img.astype(np.float32) / 255.0  # Normalize
    return img


img1 = prepare_img("1.png")
img2 = prepare_img("2.png")
img3 = prepare_img("3.png")

embedding1 = session.run(None, {"x": img1})[0][0]
embedding2 = session.run(None, {"x": img2})[0][0]
embedding3 = session.run(None, {"x": img3})[0][0]

print(cosine_similarity(embedding1, embedding2))
print(cosine_similarity(embedding2, embedding3))
print(cosine_similarity(embedding1, embedding3))

# Запустите инференс
start = time.time()
for i in range(0, 100):
    results = session.run(None, {"x": img})
print(time.time() - start)
print("Результаты:", len(results[0][0]))
