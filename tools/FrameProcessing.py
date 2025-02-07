import time
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from tools.face import face_postprocessing
from tools.face.face_detection import FaceDetector
from tools.face.face_embedding import FaceEmbedding
from tools.face.face_postprocessing import get_each_face_norm_image
from numpy.linalg import norm


def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


class FrameProcessing:
    def __init__(self, model_name="buffalo_l", res=640):
        self.cur_frame = None
        self.cur_outlined_frame = None
        self.tracker = DeepSort(max_age=30)
        self.found_faces = {}
        self.cur_faces = None
        self.cur_norm_faces_img = None
        self.face_db = {}
        self.next_person_id = 1
        self.face_embeddor = FaceEmbedding(model_path="mobilenetv2_050_Opset16.onnx")
        self.face_detector = FaceDetector(model_name=model_name, res=res, allowed_modules=['detection'])

    def set_frame(self, frame):
        self.cur_frame = frame

    def track(self):
        detections = []
        for face in self.cur_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            score = face.det_score
            detections.append([[x1, y1, x2, y2], score, None])
        tracked_faces = self.tracker.update_tracks(detections, frame=self.cur_frame)

        # _face_id = 0
        for track in tracked_faces:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            face_crop = self.cur_frame[y1:y2, x1:x2]
            # face_crop = self.cur_norm_faces_img[_face_id]
            # _face_id += 1

            # Если лицо детектировано, получаем эмбеддинг
            embedding = self.face_embeddor.get_embedding(face_crop)
            # print(embedding)
            if embedding is not None:
                # Проверяем, есть ли похожее лицо в базе
                matched_id = None
                for person_id, known_emb in self.face_db.items():
                    if cosine_similarity(embedding, known_emb) > 0.6:  # Порог совпадения
                        matched_id = person_id
                        break

                if matched_id is None:
                    matched_id = self.next_person_id
                    print(f"Found new face {matched_id}, {len(self.face_db)}")
                    self.face_db[matched_id] = embedding
                    self.next_person_id += 1  # Увеличиваем ID для следующего лица

    def process_frame(self):
        if self.cur_frame is None:
            raise ValueError("Frame wasn't set up. Call set_frame().")

        self.cur_faces = self.face_detector.get_faces(self.cur_frame)
        self.cur_norm_faces_img = get_each_face_norm_image(self.cur_frame, self.cur_faces)  # +- 0.001 sec
        self.track()
        self.cur_outlined_frame = face_postprocessing.outline_faces_at_img(self.cur_frame, self.cur_faces)

    def get_outlined_frame(self):
        return self.cur_outlined_frame
