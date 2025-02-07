import time
import cv2
import numpy as np
import onnxruntime


class FaceEmbedding:
    def __init__(self, model_path="resnet50-v2-7.onnx", providers=["TensorrtExecutionProvider","CUDAExecutionProvider", "CPUExecutionProvider"]):
        self.session = onnxruntime.InferenceSession(path_or_bytes=model_path, providers=providers)

    def get_embedding(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img.astype(np.float32) / 255.0  # Normalize
        embeddings = self.session.run(None, {"x": img})
        return embeddings[0][0]

