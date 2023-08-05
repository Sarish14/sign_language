# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:17:32 2022

@author: Admin
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Setup our webcam

cap = cv2.VideoCapture(0)                                                           # used to access our webcam, 0 is id for our webcam

detector=HandDetector(maxHands=1)                                               # initialised object to find hands, max=1

offset=20
imgSize=300

folder="Data/Y"
counter=0

while True:
    success, img=cap.read()
    hands, img=detector.findHands(img)                                       # to detect the hand from image
   
    if hands:                                                                # cropping the required image
        hand=hands[0]                                                           # since we have only 1 hand in frame, storing that in 'hand'
        x,y,w,h=hand['bbox']                                                 # extracting width and height of the hand
       
        imgWhite=np.ones((imgSize, imgSize, 3), np.uint8)*255                     # creating a fixed image size      
        imgCrop=img[y-offset:y+h+offset, x-offset:x+w+offset]                                # dimensions we want to crop it at
                             
       
        aspectRatio=h/w
       
        # set the cropped image to completely cover the white area
        if aspectRatio>1:               # if height is greater than width
            k=imgSize/h
            width_calculated=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(width_calculated, imgSize))
            width_gap=math.ceil((300-width_calculated)/2)                        # to center the image, push image by calulatig the width by which we need to push it
            imgWhite[:, width_gap:width_calculated+width_gap]=imgResize                   # overlay our cropped image on a white image(using its height and width info)
           
        else:               # if height is lesser than width
            k=imgSize/w
            height_calculated=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize, height_calculated))
            height_gap=math.ceil((300-height_calculated)/2)                        # to center the image, push image by calulatig the width by which we need to push it
            imgWhite[height_gap:height_calculated+height_gap, :]=imgResize
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
   
   
   
   
   
    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
    if(key==ord("s")):                      # when we press "s"key, then only we save the image
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)