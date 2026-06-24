#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:42:49 2026

@author: akshitmudgal
"""


#Note:   complete this code 

#yolo model: https://github.com/akanametov/yolo-face
import cv2
import numpy as np
from ultralytics import YOLO



from tensorflow.keras.models import load_model

#step1: model load kro

print("Loading th models...")

yolo_model = YOLO("yolov8m-face.pt")
#yolo_model = YOLO("yolov12n-face.pt")

print("YOLO MODEL LOAD HO GYA HAI")

#MODEL2

MASK_MODEL_PATH = "face_mask_model.h5"

mask_model = load_model(MASK_MODEL_PATH)


print("MOBILNET V2 MODEL BHI LOAD HO GYA HAI")

IMG_SIZE =(224,224)

PERSON_CLASS_ID = 0


LABELS = {0: "With Mask", 1: "Without Mask"}

COLORS = {0: (0,255,0), 1: (0,0, 255)}

CONFIDENCE_THRESHOLD = 0.75


#MAIN FUNCTION ALL THREE MODES VIDEO IMAGE WEBCAM

def process_frame(frame):
     #stag1 YOLO SE SARRE FACES KO DETCET KRO
     
     results = yolo_model(frame, verbose=False)
     
     result = results[0]
     
     boxes = result.boxes
     
     h, w = frame.shape[:2] #(h, w, c)
     
     person_count =0
     mask_count = 0
     no_mask_count = 0
     
     for box in boxes :
         class_id = int(box.cls[0])
         
         if class_id != PERSON_CLASS_ID:
             continue
         
         confidence = float(box.conf[0])
         
         if confidence < CONFIDENCE_THRESHOLD:
             continue
         
         x1, y1, x2, y2 = map(int, box.xyxy[0])
         
         x1 = max(0, x1)
         y1 = max(0,y1)
         x2 = min(w,x2)
         y2 = min(h,y2)
         
         person_crop = frame[y1:y2, x1:x2] #1280 * 720 ==> #100 * 130
         
         if person_crop.size == 0:
             continue
         
         img_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
         
         img_resized = cv2.resize(img_rgb, IMG_SIZE)
         
         
         img_array = img_resized.astype("float32")/ 255.0
         
         img_batch = np.expand_dims(img_array, axis=0) #(224,224,3) #(1, 224,224,3)
         
         prediction  = mask_model.predict(img_batch, verbose = 0)[0][0]
         
         
         class_idx = int(prediction > 0.7)
         
         label = LABELS[class_idx]
         
         color = COLORS[class_idx]
         
         
         if class_idx == 1:
             mask_conf = prediction * 100
         else:
             mask_conf = ( 1 - prediction ) * 100
             
         person_count += 1
         if class_idx == 0:
             mask_count += 1
         else:
             no_mask_count +=1
             
         cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
         
         #label text 
         
         label_text = f"{label}: {mask_conf: .0f}%"
         
         (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
         
         cv2.rectangle(frame, (x1, y1-text_h - 10), (x1+  text_w+ 5, y1), color, -1)
         
         cv2.putText(frame, label_text, (x1+2, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
         
         cv2.putText(frame, f" YOLO: {confidence: .0%}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
         
         
def test_video(video_path):
    
    
    print(" \n video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Video not loaded")  
        return
         
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FARME_HEIGHT))
    
    save_path  = video_path.replace(".", "_yolo_result.")
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    frame_num = 0
    print(" processing")
    
    while True:
        ret, frame, = cap.read()
        
        if not ret:
            print(" Video nahi mili | video complete ho gyi")
            break
        
        frame_num += 1
        
        result_frame = process_frame(frame)
        
        cv2.putText(result_frame, f" FRame: {frame_num}", (width-180, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200),1 
                    )
        
        cv2.imshow("YOLOv12 + MobileNetV2 - Video", result_frame)
        
        
        
        if cv2.waitKey(1) & "OxFF" in [ord("q"), 27]:
            print(" User stopped it")
            break
        
    cap.release()
    
    cv2.destroyAllWindows()
    print("video saved")
            
    

        
        
    
         
        
            
    