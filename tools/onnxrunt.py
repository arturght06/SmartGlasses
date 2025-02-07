import time

import cv2
import numpy as np
import onnxruntime

# Загрузите модель
# session = onnxruntime.InferenceSession("models/arcfaceresnet100-8.onnx", providers=['CUDAExecutionProvider'])
# session = onnxruntime.InferenceSession("models/retinaface1280.onnx", providers=["CUDAExecutionProvider"])
# session = onnxruntime.InferenceSession("models/resnet50-v2-7.onnx", providers=["CUDAExecutionProvider"])
session = onnxruntime.InferenceSession("../../retina_manually/buffalo_l/w600k_r50.onnx", providers=["CUDAExecutionProvider"])
session.set_providers(["CUDAExecutionProvider"])

print("Используемый Execution Provider:", session.get_providers())
print("Доступные Execution Providers:", onnxruntime.get_available_providers())

# Посмотрите выходные имена и формы
outputs = session.get_outputs()
for output in outputs:
    print("Имя:", output.name)
    print("Тип:", output.type)
    print("Форма:", output.shape)



# image_path = "images/crowd.png"
image_path = "images/alone.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))

cv2.imshow("Image", img)
# cv2.waitKey(0)
img = np.transpose(img, (2, 0, 1))  # HWC to CHW
img = np.expand_dims(img, axis=0)   # Add batch dimension
img = img.astype(np.float32) / 255.0  # Normalize

# Запустите инференс
start = time.time()
for i in range(0, 100):
    results = session.run(None, {"input.1": img})
print(time.time() - start)
print("Результаты:", results)
