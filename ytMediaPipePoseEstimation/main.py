# IMPORTS
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# SETTING UP THE VIDEO FEED
cap = cv2.VideoCapture(0)

# setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor image to RGB
        # OpenCV will give the image in BGR, so we need to convert it so mp understands
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection (results arr)
        results = pose.process(image)

        # print(results)
        # print(results.pose_landmarks)
        # print(mp_pose.POSE_CONNECTIONS)

        # Recolor image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections
        print(mp_drawing.draw_landmarks)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # dot
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) #line
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
