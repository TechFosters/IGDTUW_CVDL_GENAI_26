#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:09:42 2026

@author: akshitmudgal
"""

#MAVCHINE IDENTIFY
#WHAT  COLOR AN OBJECT IS AND WHERE IT IS

#R,g, b, y, o, w, Q

import cv2
import numpy as np

COLOUR_RANGES = {
    'RED': {
        'ranges': [
            (np.array([0,   120, 70]),  np.array([10,  255, 255])), #red at lower end
            (np.array([170, 120, 70]),  np.array([179, 255, 255])), #red at higher end
        ],
        'bgr': (0, 0, 220),
        'label': 'RED object',
    },
    'GREEN': {
        'ranges': [
            (np.array([40, 70, 70]), np.array([80, 255, 255])),
        ],
        'bgr': (0, 200, 0),
        'label': 'GREEN object',
    },
    'BLUE': {
        'ranges': [
            (np.array([100, 80, 70]), np.array([130, 255, 255])),
        ],
        'bgr': (220, 0, 0),
        'label': 'BLUE object',
    },
    'YELLOW': {
        'ranges': [
            (np.array([20, 100, 100]), np.array([35, 255, 255])),
        ],
        'bgr': (0, 220, 220),
        'label': 'YELLOW object',
    },
    'ORANGE': {
        'ranges': [
            (np.array([10, 100, 100]), np.array([20, 255, 255])),
        ],
        'bgr': (0, 140, 255),
        'label': 'ORANGE object',
    },
    'WHITE': {
        'ranges': [
            (np.array([0, 0, 200]), np.array([179, 40, 255])),
        ],
        'bgr': (180, 180, 180),
        'label': 'WHITE object',
    },
}

#open camera

camera = cv2.VideoCapture(0)

print(camera)

if not camera.isOpened():
    print("Copuld not open the camrra")
    exit()
    
print("Camera accessed")
print("\n COLOUR DETECTION TOOL - PRESS A KEY TO SWITCH TO TARGET COLOUR")
print("R - RED      G - GREEN    B- BLUE" )
print("Y - YELLOW   O- ORAnge      w - whitye")
print(" Q - quit\n")
print(" Hold a coloured object in front of camera")

current_color = 'RED'
print(f"Detecting : {current_color}")

def get_colour_mask(hsv_frame, colour_name):
    info = COLOUR_RANGES[colour_name]
   
    mask = None
    
    for(low, high) in info['ranges']:
        
        m = cv2.inRange(hsv_frame, low, high)
        
        mask = m if mask is None else cv2.bitwise_or(mask, m)

    return mask


#cv pipeline

while True:
    
    success, frame = camera.read()
    if not success:
        
        break

    #bgr -> hsv
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = get_colour_mask(hsv, current_color)
    
    kernel = np.ones((5,5), np.uint8)
    
    #OPENING
    #EROSION -> SHRINKS WHITE REGION(TINY SPECKS OF NOISE)
    #DILATION -> GROWS WHITE REGION BACK(FILLS GAPS, REAL LOOK TO THE OBJECT)
    mask = cv2.erode(mask, kernel, iterations = 2)
    mask = cv2.dilate(mask, kernel, iterations = 2)
    
    
    #CONTOUR DETECTION
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    colour_info = COLOUR_RANGES[current_color]
    box_color = colour_info['bgr']
    obj_label = colour_info['label']
    
    detected_count = 0
    
    for contour in contours:
        
        area = cv2.contourArea(contour)
        
        if area < 1500:
            continue
        detected_count+=1
        
        x,y,w,h = cv2.boundingRect(contour)
        
        
        #draw thw bounding boc
        
        cv2.rectangle(frame, (x,y),(x+w,y+h), box_color, 2)
        
        cx = x+w //2
        cy = y+h //2
        
        
        cv2.circle(frame, (cx,cy), 5, box_color, -1)
        
        area_label = f"{obj_label} pos:({cx},{cy}) area:{int(area)} px"
        cv2.putText(frame, area_label, (x,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
    cv2.rectangle(frame, (0,0), (frame.shape[1], 80), (0,0,0), -1)
    
    status = f" Detecting: {current_color} | objects_found: {detected_count}"
    
    cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
    cv2.putText(frame, "R G B Y O W Q", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(180,180,180), 1)
    
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    cv2.putText(mask_3ch, "MASK MACHINE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(200,200,200),3)
    
    
    h_frame = frame.shape[0]
    mask_resized= cv2.resize(mask_3ch, (frame.shape[1]//2, h_frame))
    frame_half= cv2.resize(frame, (frame.shape[1]//2, h_frame))
    
    combined = np.hstack([frame_half, mask_resized])
    
    cv2.imshow("COLOUR DETECTION LEFT CAMERA VIEW RIGH MASK", combined)
    
    
    
    #key habdling
    
    key = cv2.waitKey(1) & 0xFF
    #BUG2: ITS current_color and not current_colour
    if key == ord('q') or key == ord('Q'): break    
    elif key == ord('r') or key == ord('R'): current_colour ='RED'; print("Detecting COLOR RED");
    elif key == ord('g') or key == ord('G'): current_colour ='GREEN'; print("Detecting COLOR GREEN");
    elif key == ord('b') or key == ord('B'): current_colour ='BLUE'; print("Detecting COLOR BLUE");
    elif key == ord('y') or key == ord('Y'): current_colour ='YELLOW'; print("Detecting COLOR YELLOW");
    elif key == ord('o') or key == ord('O'): current_colour ='ORANGE'; print("Detecting COLOR ORANGE");
    elif key == ord('w') or key == ord('W'): current_colour ='WHITE'; print("Detecting COLOR WHITE");
    
    #BUG1: THIS SHOULD BE OUT OF THE LOOP
    camera.release()
    cv2.destroyAllWindows()