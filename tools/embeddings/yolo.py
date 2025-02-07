import cv2
import numpy as np

# Загрузка модели YOLOv8-Face
net = cv2.dnn.readNetFromONNX("YOLOv8/yolov8m-face-lindevs.onnx")

# Загрузка изображения
image = cv2.imread("alone.jpg")
h, w = image.shape[:2]

# Подготовка изображения
blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
net.setInput(blob)
print(blob.shape)
# Запуск модели
detections = net.forward()
print(detections.size)
# Обработка результатов
for detection in detections[0]:
    confidence = detection[4]
    if confidence > 0.5:
        x1, y1, x2, y2 = (detection[:4] * [w, h, w, h]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("YOLOv8-Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
