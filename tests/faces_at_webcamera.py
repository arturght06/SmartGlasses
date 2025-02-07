import cv2
import time
from sort import Sort  # Подключаем SORT
from tools.FrameProcessing import FrameProcessing
from tools.face.face_detection import FaceDetector


# Инициализация трекера SORT
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 3)
frame_processing = FrameProcessing()

if not camera.isOpened():
    print("Error: Camera not accessible")
else:
    print("Camera works!")
    while True:
        ret, frame = camera.read()    # around 0.008 seconds
        if ret:
            start = time.time()

            frame_processing.set_frame(frame)
            frame_processing.process_frame()
            outlined_frame = frame_processing.get_outlFained_frame()
            each_face_images = frame_processing.cur_norm_faces_img
            if each_face_images is not None and len(each_face_images) > 0:
                cv2.imshow("face", each_face_images[0])

            print(f"{int(1/(time.time() - start))}fps. Обработка кадра заняла: {round(time.time() - start, 4)}")
            cv2.imshow('Test', outlined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Error reading frame")
            break
    camera.release()
    cv2.destroyAllWindows()
