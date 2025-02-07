import cv2
from insightface.app import FaceAnalysis


class FaceDetector:
    def __init__(self, model_name="buffalo_sc", accurancy_threshold=0.5, res=640, allowed_modules=['detection']):
        assert res in [320, 480, 640]
        # Инициализация модели RetinaFace с backbone MobileNetV3
        self.app = FaceAnalysis(
            det_face=model_name,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            allowed_modules=allowed_modules,
        )
        self.app.prepare(
            ctx_id=0,
            det_size=(res, res),  # уменьшаем разрешение для скорости 320, 480, 640
            det_thresh=accurancy_threshold,  # порог уверенности
        )

    def get_faces(self, image):
        faces = self.app.get(image)
        return faces


