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
         
     return frame
         
 #MODE 1        
def test_image(img_path):
    #Single image pe test karo
    print(f"\ Image: {img_path}")

    frame = cv2.imread(img_path)
    if frame is None:
        print("❌ Image nahi mili! Path check karo.")
        return

    # Process karo
    result_frame = process_frame(frame)

    # Dikhao
    cv2.imshow("YOLOv8 + MobileNetV2 — Image", result_frame)
    print("Koi bhi key dabao band karne ke liye...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save karo
    save_path = img_path.replace(".", "_yolo_result.")
    cv2.imwrite(save_path, result_frame)
    print(f" Saved: {save_path}")



# MODE 2: VIDEO


def test_video(video_path):
    #Video file pe test karo
    print(f"\n Video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(" Video nahi mili! Path check karo.")
        return

    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"   {width}*{height} | {fps} FPS | {total} frames")

    # Output video
    save_path = video_path.replace(".", "_yolo_result.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    frame_num = 0
    print("   Processing... (Press Q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Video complete!")
            break

        frame_num += 1

        # Har frame pe process karo
        result_frame = process_frame(frame)

        # Frame number dikhao
        cv2.putText(result_frame, f"Frame: {frame_num}/{total}",
                    (width - 180, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        # Screen pe dikhao
        cv2.imshow("YOLOv8 + MobileNetV2 — Video", result_frame)

        # Output mein save karo
        out.write(result_frame)

        # Q = quit
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            print("   User ne band kiya.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f" Result saved: {save_path}")



# MODE 3: LIVE WEBCAM


def test_webcam(camera_index=0):
    #Live webcam se real-time detection
    print(f"\ Webcam ({camera_index}) statreing...")
    print("   Q ya ESC dabao band karne ke liye")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Webcam nahi mili!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    import time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror
        frame = cv2.flip(frame, 1)

        # Process karo
        result_frame = process_frame(frame)

        # FPS calculate karo
        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time

        # FPS dikhao
        h, w = result_frame.shape[:2]
        cv2.putText(result_frame, f"FPS: {fps:.0f}",(w - 100, 35),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 255), 2)
        # Yellow mein FPS

        cv2.imshow("YOLOv8 + MobileNetV2 — Live", result_frame)

        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam band ho gayi!")


# MAIN


if __name__ == "__main__":

    print("=" * 55)
    print("   YOLOv8 + MobileNetV2 — Face Mask Detection")
    print("=" * 55)
    print()
    print("  1 :Image file")
    print("  2 : Video file")
    print("  3: Live Webcam")
    print()

    choice = input("Choice (1/2/3): ").strip()

    if choice == "1":
        path = input("Image path: ").strip()
        test_image(path)

    elif choice == "2":
        path = input("Video path: ").strip()
        test_video(path)

    elif choice == "3":
        cam = input("Camera index (0=built-in, Enter=default): ").strip()
        test_webcam(int(cam) if cam else 0)

    else:
        print(" 1, 2, ya 3 enter karo!")


