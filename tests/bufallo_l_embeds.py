import os
import time
import cv2
import numpy as np

from tools.face.face_detection import FaceDetector

dataset_path = 'faces'
face_detector = FaceDetector("buffalo_sc")


def get_file_paths(directory):
    for root, dirs, files in os.walk(directory):
        paths = [os.path.join(root, file) for file in files]
        return paths


images = []
for file_path in get_file_paths(dataset_path):
    image = cv2.imread(file_path)
    images.append(image)

batch = np.concatenate(images, axis=0)
# images = [cv2.imread("faces/crowd.png")]
start = time.time()
faces = face_detector.get_faces(images)
print(len(faces))
# for image in images:
#     faces = face_detector.get_faces(image)
#     print(len(faces))
    # frame = face_detector.outline_faces_at_img(image, faces)
    # cv2.imshow('Face', frame)
    # cv2.waitKey(0)

    # x, y, x2, y2 = faces[0].bbox.astype(int)
    # face = image[y:y2, x:x2]
    # cv2.imshow('Face', face)
    # cv2.waitKey(0)

taken_time = round(time.time() - start, 3)
per_one = round(taken_time / len(images), 5)
print("Time: ", taken_time, "For sec/img: ", per_one)

cv2.destroyAllWindows()
