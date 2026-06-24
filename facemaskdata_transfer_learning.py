#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:24:47 2026

@author: akshitmudgal
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D

from tensorflow.keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

print("Tensorflow version: ", tf.__version__)
print("Libraries load ho gyi hai")

#step2

#dataset ka path set krenge

USERNAME = "akshitmudgal"

BASE_PATH = "/Users/akshitmudgal/Desktop/IGDTUW_DIP/Face Mask Dataset"

TRAIN_PATH = os.path.join(BASE_PATH, "Train")

TEST_PATH = os.path.join(BASE_PATH, "Test")

VALIDATION_PATH = os.path.join(BASE_PATH, "Validation")

#folders exist krte hain?/
for p in [TRAIN_PATH, TEST_PATH, VALIDATION_PATH]:
    
    if os.path.exists(p):
        print(f" Found : {p}")
    else:
        print(f" Not Found : {p}")
        
#defining the constants
IMG_SIZE = (224,224)

BATCH_SIZE = 32

EPOCH = 10


#DATA GENERATORS BNANA

train_datagen = ImageDataGenerator(
    
    rescale = 1.0/255, #for normalization
    
    #for augmentation
    rotation_range = 20,
    
    zoom_range = 0.2,
    
    horizontal_flip = True,
    
    width_shift_range = 0.1,
    
    height_shift_range = 0.1,
    
    )

val_test_datagen = ImageDataGenerator(
    
    rescale = 1.0/255
    
    
    )

#training generator

train_gen = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "binary"
    )

val_gen  = val_test_datagen.flow_from_directory(
    VALIDATION_PATH,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "binary"
    
    )

test_gen= val_test_datagen.flow_from_directory(
    TEST_PATH,
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = "binary",
    
    shuffle = False
    
    )

print("\nClasses: ", train_gen.class_indices)

print(f" Train images: {train_gen.samples}")
print(f" Validation images: {val_gen.samples}")
print(f" Testing images: {test_gen.samples}")

#MOBILENETV2 LOAD KRO(PRETRAINED)

base_model = MobileNetV2(
    weights = "imagenet",
    
    include_top=False,
    
    input_shape = (224,224,3)
    )

base_model.trainable = False



print(f"\n Base model layers: {len(base_model.layers)}")
print("Base model has been frozen")
#(None, 7,7, 1280)
x = base_model.output

x = GlobalAveragePooling2D()(x)

#inout(7,7,1280) output (1280, ) #flatten 7 * 7* 1280=

x = Dense(128, activation ="relu")(x)

x = Dropout(0.5)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs= base_model.input, outputs =output)

print("Custom layers just got added")

#sgd, #rmsporp
model.compile(
    optimizer = "adam",
    loss= "binary_crossentropy",
    metrics=["accuracy"]
    )

model.summary()
#trian high, validation low 
#epoch3 val_acc = 96.0
#epoch_4 = 95.5
#epoch5 = 95.2
#epoch6 = 94.8

early_stop = EarlyStopping(monitor = "val_accuracy", patience = 3,restore_best_weights = True, verbose=1)

#model train kro
print("\n starting the training")
print("=" * 40)

history= model.fit(train_gen, epochs = EPOCH, validation_data = val_gen, callbacks=[early_stop], verbose = 1)


#history -> training ka complete rtecord
#history.history -> dictionary {loss, accuracy, val_loss, val_accuracy}

print("Traning complete")

#step training graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy graph
ax1.plot(history.history["accuracy"],     label="Train Accuracy",      color="#7F77DD")
ax1.plot(history.history["val_accuracy"], label="Validation Accuracy",  color="#1D9E75")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss graph
ax2.plot(history.history["loss"],     label="Train Loss",      color="#D85A30")
ax2.plot(history.history["val_loss"], label="Validation Loss",  color="#BA7517")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graph save ho gaya: training_results.png ✓")


#STEP 11: Test set pe evaluate karo 
print("\nTest set pe evaluate kar rahe hain...")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")
print(f"   Test Loss:     {test_loss:.4f}")


# ── STEP 12: Model save karo ─────────────────────────────────
model.save("face_mask_model.h5")
print("\nModel save ho gaya: face_mask_model.h5 ✓")
print("Isko baad mein load karke use kar sakte ho!")


