#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 09:51:16 2026

@author: akshitmudgal
"""

import cv2
import matplotlib.pyplot as plt

camera = cv2.VideoCapture(0)

print("Camera Object", camera)


if not camera.isOpened():
    print("Error: Could not find the camera")
    print("Make sure webcam is connected and not being used by another app")
    exit()
    
#FPS
fps = camera.get(cv2.CAP_PROP_FPS)
print("FPS = ", fps)
print("Camera opened successfully")

print("Camera is running...")
print(" Press Q in the video window to quit")
print(" Press S to save (snapshot) of the current frame.")

#read camera props

frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) #wide
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) #TALL
fps = int(camera.get(cv2.CAP_PROP_FPS)) #FRAMES PER SECOND

print(f"\n Camera resolution: {frame_width} * {frame_height} pixels")
print(f" Frames per second: {fps} FPS")

snapshot_count = 0

while True: 
    
    #frame BGR format
    success, frame = camera.read()
    
    if not success:
        print("Failed to grab the frame, thus stopping")
        break
    
    #overlaying
        #IMAGE, TEXT, POSN, FONT, FS, COLOR, THICKNESS
    cv2.putText(frame, "Press Q to quit | S for snapshot", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    frame_number = int(camera.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.putText(frame, f"Frame: {frame_number }", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    cv2.imshow("LIVE CAMERA FEED - CV01", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("Q pressed-> quitting now")
        break
    elif key == ord('s') or key == ord('S'):
        snapshot_count += 1
        filename = f"snapshot_{snapshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Snapshot saved: {filename}")
        
camera.release()
cv2.destroyAllWindows()

print("Camera released")

