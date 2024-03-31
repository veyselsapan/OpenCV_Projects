import cv2
import mediapipe as mp
import time
import numpy as np


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, 
                                        max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.detectionCon, 
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # converting the image to RGB because the hands object only accepts RGB images
        self.results = self.hands.process(imgRGB) # processing the image#print(results.multi_hand_landmarks)  # printing the landmarks of the hand
        if self.results.multi_hand_landmarks:  # if there are hands in the image
            for handLms in self.results.multi_hand_landmarks:  # for each hand in the image
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # drawing the connections between the landmarks
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []  # list to store the landmarks
        if self.results.multi_hand_landmarks:  # if there are hands in the image
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # for each landmark in the hand
                    h, w, c = img.shape  # getting the height, width and channel of the image
                    cx, cy = int(lm.x*w), int(lm.y*h)  # getting the x and y coordinates of the landmark
                    xList.append(cx)
                    yList.append(cy)
                    self.lmList.append([id, cx, cy])  # appending the id and the coordinates of the landmark to the list
                    if draw:  # if the id is 0, then it is the first landmark
                        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # drawing a circle at the landmark
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0, 255, 0), 2)
        return self.lmList, bbox
    
    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.lmList[id*4][2] < self.lmList[id*4-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = ((x2-x1)**2 + (y2-y1)**2)**0.5
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector() # creating an object of the HandDetector class
    while True:
        success, img = cap.read()  # reading the image from the webcam
        img = detector.findHands(img) # finding the hands in the image
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()  # getting the current time
        fps = 1/(cTime-pTime)  # calculating the frames per second
        pTime = cTime  # setting the previous time to the current time
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)  # putting the fps on the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()