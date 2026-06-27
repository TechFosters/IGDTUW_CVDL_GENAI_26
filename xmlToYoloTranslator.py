#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:44:36 2026

@author: akshitmudgal
"""

#translator

import os, shutil, random
import xml.etree.ElementTree as ET
from pathlib import Path



SRC_IMAGES = "helmet_dataset/images"
SRC_ANNOATIONS = "helmet_dataset/annotations"
OUT_DIR = "helmet_yolo"
TRAIN_RATIO = 0.8 #80% train #20 % validation
SEED = 42


CLASS_MAP = {
    "with helmet": 0,
    "without helmet": 1,
    "With Helmet": 0,
    "Without Helmet": 1,
    "helmet": 0,
    "no helmet": 1,
    "head": 1
    
    }

random.seed(SEED)

print("+"* 60)
print("XML TO YOLO CONVERTER")


#SECTION2

for split in ["train", "val"]:
    os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok =True);
    os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok =True);

#section 3
xml_files = sorted(Path(SRC_ANNOATIONS).glob("*.xml"))

random.shuffle(xml_files)

split_idx = int(len(xml_files) * TRAIN_RATIO)
train_xmls= xml_files[:split_idx]
val_xmls= xml_files[split_idx:]

print(f"Train : {len(train_xmls)} | Validation: {len(val_xmls)}")

#section 4

def convert_xml(xml_path, split):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    
    
    if img_w == 0 or img_h == 0:
        return False, "zero size"  #corrupted annotation
    
    
    filename = root.find("filename").text
    img_src = None
    
    
    for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
        candidate = os.path.join(SRC_IMAGES, os.path.splitext(filename)[0] + ext)
        if os.path.exists(candidate):
            img_src = candidate
            break
    if img_src is None:
        return False, f"Image not founf: {filename}"
    
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        cls = CLASS_MAP.get(name)
        
        if cls is None:
            print(f" Unknown class '{name}' in {xml_path.name} ")
            
        bndbox = obj.find("bndbox")
        
        xmin = float(bndbox.find("xmin").text)
        ymin= float(bndbox.find("ymin").text)
        xmax= float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        xmin = max(0, min(xmin,img_w))
        xmax = max(0, min(xmax,img_w))
        ymin = max(0, min(ymin,img_h))
        ymax = max(0, min(ymax,img_h))
        
        if xmax<= xmin or ymax<=ymin:
            continue
        
        cx = ((xmin+xmax)/2) /img_w
        cy = ((ymin + ymax)/2)/img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        
        #yolo line format: class_id cx cy bw bh
        lines.append(f" {cls} {cx:.6f} {cy: .6f} {bw:.6f} {bh:.6f}")
    if not lines:
        return False, "no valid object"
    
    stem = Path(img_src).stem
    
    img_dst = f"{OUT_DIR}/images/{split}/{stem}{Path(img_src).suffix}"
    shutil.copy2(img_src, img_dst)
    
    lbl_dst = f"{OUT_DIR}/labels/{split}/{stem}.txt"
    with open(lbl_dst, "w") as f:
        f.write("\n" .join(lines))
        
    return True, len(lines)

stats = {"train": {"ok": 0, "skip":0}, "val":{"ok": 0, "skip":0}}
         
for split, xmls in [("train", train_xmls), ("val", val_xmls)]:
    print(f"\n Converting {split}....")
    for xml_path in xmls:
        ok, info = convert_xml(xml_path, split)
        if ok: stats[split]["ok"] += 1
        else: stats[split]["skip"] +=1
        
        
yaml_content = f"""# YOLOv8 Helmet Detection Datset config #data kahan hai aur classe skya hai

path: {os.path.abspath(OUT_DIR)}
train: images/train
val: images/val

nc: 2
names:
    0: with_helmet
    1: without_helmet

"""
         

yaml_path = f"{OUT_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)
print("data.yaml wriiten")