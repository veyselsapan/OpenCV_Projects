import cv2
import mediapipe as mp
import time


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
        lmList = []  # list to store the landmarks
        if self.results.multi_hand_landmarks:  # if there are hands in the image
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):  # for each landmark in the hand
                    h, w, c = img.shape  # getting the height, width and channel of the image
                    cx, cy = int(lm.x*w), int(lm.y*h)  # getting the x and y coordinates of the landmark
                    lmList.append([id, cx, cy])  # appending the id and the coordinates of the landmark to the list
                    if draw:  # if the id is 0, then it is the first landmark
                        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)  # drawing a circle at the landmark
        return lmList
    

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