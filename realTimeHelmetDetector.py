#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 13:04:11 2026

@author: akshitmudgal
"""

#COMPLETE THIS CODE STUDENTS, FIX TEH SYNTAX ERROPR IF ANY
#JUST NEED TO ADD HOW TO TEST ON IMAGE, VIDEO AND WEBCAM



import os, cv2, time, argparse
from datetime import datetime
from ultralytics import YOLO


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="webcam")
parser.add_argument("--save", action="store_true")
parser.add_argument("--conf", type=float, default = 0.35)
args = parser.parse_args()

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "yolo_helmet_training", "hemlet_v1", "weights", "best.pt")

COLORS = {
    "with_helmet": (0,210,0),
    
    "without_helmet": (0,0,220)
    }

os.makedirs(OUTPUT_DIR, exist_ok = True)

#SOURCE TYPE
src = args.source.strip()
IMAGE_EXTS= ('.jpg', '.jpeg', '.png', 'bmp', 'webp')
VIDEO_EXTS = ('.mp4', 'avi', '.mov', '.mkv', '.webm')

if src == "webcam":
    MODE = "webcam"
elif src.lower().endswith(IMAGE_EXTS): MODE = "image"
elif src.lower().endswith(VIDEO_EXTS): MODE = "video"
else:
    MODE = "webcam"
    
print("+"* 60)
print(f"YOLOV8n HELMET DETECTOR | {MODE.upper()}")

#LOAD MODEL
if not os.path.exists(MODEL_PATH):
    print("Trained model nahi mila")
    print("Pehle helmet_yolo_trained.py file run karo")
    raise SystemExit
    
    
    
print("\n Loading traiend yolo model")
model = YOLO(MODEL_PATH)
print(f" Model LOaded: {MODEL_PATH}")
print(f" Classes: {model.names}")


def put_label(img, text, x, y, color):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y = max(th + bl + 4, y)
    cv2.rectangle(img, (x,y-th-bl-2), (x+tw+6, y+bl), color, cv2.FILLED)
    lum = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
    tc = (0,0,0) if lum > 130 else (255,255,255)
    cv2.putText(img, text, (x+3, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2)
    

def process_frame(frame, conf, debug=False):
    results = model(frame, conf = conf, verbose=False)
    helmet_n, no_helmet_n = 0,0

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf_val = float(box.xonf[0])
            color = COLORS.get(cls_name, (128,128,128))
            
            
            #box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            
            
            #label
            label = f"{'yes'if cls_name == "with_helmet" else'NO'} {cls_name.replace('_','').title()} {conf_val*100:.0f}%"
            put_label(frame, label, x1, y1-5, color)
            
            if debug:
                cv2.putText(frame, f"cls: {cls_id}", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255),1 )
            
            if cls_name == "with_helmet":   helmet_n+=1
            else:                           no_helmet_n+=1
    return frame, helmet_n, no_helmet_n
    