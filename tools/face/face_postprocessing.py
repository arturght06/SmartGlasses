import cv2
from insightface.utils import face_align


def get_each_face_norm_image(image, faces):
    """
    Для каждого обнаруженного лица возвращает выровненное изображение лица.

    Если у лица доступны 5 ключевых точек (атрибут 'kps'), выполняется выравнивание
    с помощью face_align.norm_crop, иначе лицо вырезается по координатам bounding box.

    Аргументы:
        image (numpy.ndarray): Исходное изображение.
        faces (list): Список объектов лица, полученных, например, через insightface.app.FaceAnalysis.get().

    Возвращает:
        list: Список изображений лиц.
    """
    face_imgs = []
    for face in faces:
        # Если доступны 5 ключевых точек, выполняем выравнивание
        if hasattr(face, "kps") and face.kps is not None:
            # norm_crop выравнивает лицо на основании координат landmark
            aligned_face = face_align.norm_crop(image, landmark=face.kps)
            face_imgs.append(aligned_face)
        else:
            # Если ключевые точки не обнаружены, используем обычное вырезание по bbox
            x, y, x2, y2 = face.bbox.astype(int)
            face_imgs.append(image[y:y2, x:x2])
    return face_imgs


def outline_faces_at_img(image, faces):
    """
    Отрисовывает прямоугольники (bounding box) вокруг каждого обнаруженного лица.

    Аргументы:
        image (numpy.ndarray): Исходное изображение.
        faces (list): Список объектов лица, содержащих координаты bbox.

    Возвращает:
        numpy.ndarray: Изображение с отрисованными прямоугольниками вокруг лиц.
    """
    for face in faces:
        x, y, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(image, (x, y), (x2, y2), (0, 255, 0), 2)
    return image
