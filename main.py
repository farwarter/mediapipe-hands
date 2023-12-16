import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands.Hands(static_image_mode=False,
                         max_num_hands=1,
                         min_tracking_confidence=0.5,
                         min_detection_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

while True:
    ret, img = cap.read()
    result = hands.process(img)
    print(result)
    if result.multi_hand_landmarks:
        for id, lm in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 255))
            #print(id)
            #print(lm)
        mpDraw.draw_landmarks(img, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("Hand tracking", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
