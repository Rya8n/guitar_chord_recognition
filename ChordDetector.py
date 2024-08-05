import cv2
import time
import math
import pickle
import numpy as np
import HandTrackingModule as htm
import tensorflow as tf
from tensorflow import keras

#wCam, hCam = 640, 480
#wCam, hCam = 848, 480
wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(detectionCon=1)
loaded_model = keras.models.load_model("deltaNET.h5")
loaded_labels = pickle.load(open("labels.dat", "rb"))
chord = "X"

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    keypoints, landmarks = detector.findPosition(img, draw=False)

    if len(landmarks) != 0:
        kptList = np.array([landmarks], dtype=np.float32)
        ### Making prediction
        pred = loaded_model.predict(kptList)
        chord = loaded_labels[pred.argmax()]


        print(chord)
        cv2.putText(img, str(chord), (20, 450), cv2.FONT_HERSHEY_PLAIN,
                    6, (255, 0, 0), 5)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #cv2.putText(img, f'FPS: {int(fps)}', (500, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    cv2.namedWindow('custom window', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("custom window", img)
    cv2.resizeWindow("custom window", 864, 480) 
    cv2.waitKey(1)