import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

class HandAuthenticator:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    # -------------------------------
    # Hand detection
    # -------------------------------
    def detect_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label  # "Left" / "Right"

        return hand_landmarks, handedness

    def extract_landmarks(self, hand_landmarks):
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    # -------------------------------
    # Palm orientation detection
    # -------------------------------
    def is_palm_front(self, landmarks):
        WRIST = 0
        INDEX_MCP = 5
        PINKY_MCP = 17

        wrist = landmarks[WRIST]
        index_mcp = landmarks[INDEX_MCP]
        pinky_mcp = landmarks[PINKY_MCP]

        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        normal = np.cross(v1, v2)

        # MediaPipe camera looks toward -Z
        return normal[2] < 0

    # -------------------------------
    # Gesture classification (STATE BASED)
    # -------------------------------
    def classify_gesture(self, landmarks, handedness):
        if landmarks is None:
            return None

        palm_front = self.is_palm_front(landmarks)

        # -------------------------------
        # Finger openness (Y-axis, stable)
        # -------------------------------
        index_open  = landmarks[8][1]  < landmarks[6][1]
        middle_open = landmarks[12][1] < landmarks[10][1]
        ring_open   = landmarks[16][1] < landmarks[14][1]
        pinky_open  = landmarks[20][1] < landmarks[18][1]

        fingers = [index_open, middle_open, ring_open, pinky_open]
        count = fingers.count(True)

        # -------------------------------
        # THUMB LOGIC — EXPLICIT STATES
        # -------------------------------

        # Right hand
        if handedness == "Right":
            if palm_front:
                # RF: thumb points LEFT when open
                thumb_open = landmarks[4][0] < landmarks[3][0]
                state = "RIGHT_FRONT"
            else:
                # RB: thumb points RIGHT when open
                thumb_open = landmarks[4][0] > landmarks[3][0]
                state = "RIGHT_BACK"

        # Left hand
        else:
            if palm_front:
                # LF: thumb points RIGHT when open
                thumb_open = landmarks[4][0] > landmarks[3][0]
                state = "LEFT_FRONT"
            else:
                # LB: thumb points LEFT when open
                thumb_open = landmarks[4][0] < landmarks[3][0]
                state = "LEFT_BACK"

        # -------------------------------
        # GESTURE DECISION
        # -------------------------------
        # (Same logic, but thumb meaning is now correct per state)

        
        if handedness == "Right":
            if count == 4 and not thumb_open:
                return "OPEN_PALM"
            
            if count == 4 and thumb_open:
                return "FOUR_FINGERS"
            
            if count == 0 and thumb_open:
                return "FIST"
        
        else:
            if count == 4 and thumb_open:
                return "OPEN_PALM"

            if count == 4 and not thumb_open:
                return "FOUR_FINGERS"

            if count == 0 and not thumb_open:
                return "FIST"

        if index_open and middle_open and not ring_open and not pinky_open:
            return "TWO_FINGERS"

        return "UNKNOWN"
