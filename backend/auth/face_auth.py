import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

class FaceAuthenticator:
    def __init__(self):
        self.face_detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )

    def detect_face(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)

        if not results.detections:
            return None,None

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w, _ = frame.shape

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = x1 + int(bbox.width * w)
        y2 = y1 + int(bbox.height * h)
        
        # 🔐 Clamp bounding box to frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None, None

        face = frame[y1:y2, x1:x2]
        
        if face.size == 0:
            return None, None

        return face, (x1, y1, x2, y2)

    def extract_embedding(self, face):
        if face is None or face.size == 0:
            return None
        
        resized = cv2.resize(face, (100, 100))
        embedding = resized.flatten().astype("float32")
        embedding /= (np.linalg.norm(embedding) + 1e-10)
        return embedding
