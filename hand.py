import mediapipe as mp
import numpy as np
import cv2
import math


class HandTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLm in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

    def getPostion(self, img, handNo = 0, draw=True):
        lmList =[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for lm in myHand.landmark:
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append((cx, cy))

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,255), cv2.FILLED)
        return lmList
    
    def getUpFingers(self, img):
        pos = self.getPostion(img, draw=False)
        self.upfingers = []
        if pos:
            #thumb
            self.upfingers.append((pos[4][1] < pos[3][1] and (pos[5][0]-pos[4][0]> 10)))
            #index
            self.upfingers.append((pos[8][1] < pos[7][1] and pos[7][1] < pos[6][1]))
            #middle
            self.upfingers.append((pos[12][1] < pos[11][1] and pos[11][1] < pos[10][1]))
            #ring
            self.upfingers.append((pos[16][1] < pos[15][1] and pos[15][1] < pos[14][1]))
            #pinky
            self.upfingers.append((pos[20][1] < pos[19][1] and pos[19][1] < pos[18][1]))
        return self.upfingers

    def get_distance(self, point1_idx, point2_idx, frame, draw=False):
        lmList = self.getPostion(frame, draw=False)
        if lmList:
            x1, y1 = lmList[point1_idx][0], lmList[point1_idx][1]
            x2, y2 = lmList[point2_idx][0], lmList[point2_idx][1]
            distance = math.hypot(x2 - x1, y2 - y1)
            if draw:
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(frame, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            return distance
        return None