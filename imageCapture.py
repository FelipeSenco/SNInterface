import cv2
import mediapipe as mp
import pyautogui as pygui
from collections import deque
from enum import Enum


class Mode(Enum):
    TRACKING = 1
    JOYSTICK = 2


# Initialize MediaPipe Hand solution
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize MediaPipe Drawing utility
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")


# Initialization
mode = Mode.TRACKING
thumbIsRight = False
previous_wrist_landmark = None
scaling_factor_x = scaling_factor_y = 1000
pygui.FAILSAFE = False
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Moving average parameters
N = 5  # Number of frames to include in the moving average
dx_buffer = deque(maxlen=N)
dy_buffer = deque(maxlen=N)


def check_thumbs_up(hand_landmarks):
    global thumbIsRight
    # check thumbs up
    if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
        if not thumbIsRight:
            thumbIsRight = True
            pygui.mouseUp()
            print("Mouse Up")
    else:
        if thumbIsRight:
            thumbIsRight = False
            print("Mouse Down")
            pygui.mouseDown()


def track_wrist_movement(hand_landmarks):
    global previous_wrist_landmark, dx_buffer, dy_buffer
    # check wrist movement
    if previous_wrist_landmark is not None:
        dx = hand_landmarks.landmark[0].x - previous_wrist_landmark.x
        dy = hand_landmarks.landmark[0].y - previous_wrist_landmark.y

        dx_buffer.append(dx)
        dy_buffer.append(dy)

        dx_avg = sum(dx_buffer) / N
        dy_avg = sum(dy_buffer) / N

        pygui.moveRel(-dx * scaling_factor_x, dy * scaling_factor_y)

    previous_wrist_landmark = hand_landmarks.landmark[0]


try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(rgb_image)

        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                check_thumbs_up(hand_landmarks)
                if mode == Mode.TRACKING:
                    track_wrist_movement(hand_landmarks)
                elif mode == Mode.JOYSTICK:
                    pass

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
