import cv2
import numpy as np

# Загрузка модели RetinaFace в формате ONNX
net = cv2.dnn.readNetFromONNX("retinaface-resnet50.onnx")

# Загрузка изображения
image = cv2.imread("alone.jpg")
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Запуск модели
detections = net.forward()

# Разбор выходных данных
bboxes = detections[0]  # Первый выходной тензор содержит координаты

# Обработка всех найденных лиц
for bbox in bboxes:
    confidence = bbox[4]  # Уверенность модели
    if confidence > 0.5:  # Фильтрация слабых детекций
        x1, y1, x2, y2 = (bbox[:4] * [w, h, w, h]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("RetinaFace Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()