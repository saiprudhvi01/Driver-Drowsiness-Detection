import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import time
import threading
import winsound  # For system beep on Windows

# EAR Calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# System beep alert function
def sound_alert():
    duration = 1000  # milliseconds
    freq = 1000      # Hz
    winsound.Beep(freq, duration)

# Thresholds
thresh = 0.25
frame_check = 3  # seconds
flag = 0
start_time = None
alerted = False

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye indices in MediaPipe (for left and right eye)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            leftEAR = eye_aspect_ratio(left_eye)
            rightEAR = eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw eye contours
            cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)

            if ear < thresh:
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time

                if elapsed_time >= frame_check and not alerted:
                    threading.Thread(target=sound_alert).start()
                    alerted = True
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 460),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                start_time = None
                alerted = False

    cv2.imshow("Driver Drowsiness Detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
