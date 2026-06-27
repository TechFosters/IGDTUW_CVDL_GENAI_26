#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:47:12 2026

@author: akshitmudgal
"""

from ultralytics import YOLO
import os

DATA_YAML = "helmet_yolo/data.yaml"
OUTPUT_DIR = "outputs"
PROJECT =os.path.join(OUTPUT_DIR, "yolo_helmet_training")
RUN_NAME = "helmet_v1"

print("+"* 60)
print("YOLOV8n training")

model = YOLO('yolov8n.pt')

result = model.train(
    data = DATA_YAML,
    epochs = 50,
    imgsz = 640,
    batch = 16,
    patience = 10,
    project = PROJECT,
    name = RUN_NAME,
    device = "mps", #cuda
    workers= 4, 
    exist_ok =True,
    
    flipud = 0.1,
    fliplr = 0.5,
    mosaic = 1.0,
    degrees = 10.0,
    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
        
    )
print("+"* 60)
print("YOLOV8n training complete")

best_model = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
