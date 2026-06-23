#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:00:27 2026

@author: akshitmudgal
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("    HANDWRIITEN DIGIT RECOGNIZER - CNN MINI PROJECT")
print("=" * 65)

#LOAD THE DATA
print("\n STEP1 : LOADING MNIST DATASET...")
print("=" * 65)

(X_train, y_train), (X_test, y_test)= keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape} --> {X_train.shape[0]} images, each {X_train.shape[1]} * {X_train.shape[2]} pixels")

print(f"Trainng Labels: {y_train.shape} --> {y_train.shape[0]} numbers (0-9)")

print(f"Test images : {X_test.shape}")

print(f"Pixel Range : [{X_train.min()}, [{X_train.max()}]")

X_train = X_train/255.0
X_test = X_test/255.0

print("\n STEP2 : reshape")
print("=" * 40)

X_train_cnn = X_train.reshape(-1,28,28,1)
X_test_cnn = X_test.reshape(-1,28,28,1)



print("\n STEP3 : BUILDING THE CNN ARCHITECTURE...")
print("=" * 40)

cnn_model = keras.Sequential([
    
    
    layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)), 
    
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    
    #flaten
    
    layers.Flatten(),
    
    #classifier
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
    
    
    
    ])

cnn_model.summary()


print("\n STEP4 : compiling and training")
print("=" * 40)

cnn_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])

print("Optimizer: Adam | loss: s_c_crosentropy| metrics accuracy")

print("\n STEP 5Training ...(~1-3 minutes depending upon ur hardware_ \n")

cnn_history = cnn_model.fit(
    X_train_cnn, #input 60000, 28, 28, 1
    y_train, # 7 hai 
    
    epochs = 5,
    
    validation_data=(X_test_cnn, y_test),
    #agar train_accuracy > val_acc = OVERFITTING
    verbose =1 # o kuch mat print karo, 
    )


print("\n STEP 6 : tarining ann  fro compariosn")
print("=" * 40)

ann_model = keras.Sequential([
    layers.Flatten(input_shape=(28,28)),
    
    layers.Dense(128, activation = 'relu'),
    
    layers.Dense(10, activation = 'softmax')
    
    ])

ann_model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])


ann_history = ann_model.fit(
    X_train, #input 60000, 28, 28, 1
    y_train, # 7 hai 
    
    epochs = 5,
    
    validation_data=(X_test, y_test),
    #agar train_accuracy > val_acc = OVERFITTING
    verbose =1 # o kuch mat print karo, 
    )

print("\n STEP 7: comparison")
print("=" * 40)

ann_test_loss, ann_test_acc = ann_model.evaluate(X_test, y_test, verbose = 0)
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test_cnn, y_test, verbose = 0)

print(f"\n ANN KI TEST ACCURACY: {ann_test_acc * 100: .2f}%")

print(f"\n CNN KI TEST ACCURACY: {cnn_test_acc * 100: .2f}%")

print(f"\n Differnce: {(cnn_test_acc - ann_test_acc)*100:.2f}")

plt.figure(figsize = (10,4))

plt.subplot(1,2,1)

plt.plot(ann_history.history['val_accuracy'], label ='ANN',marker= 'o')
plt.plot(cnn_history.history['val_accuracy'], label ='CNN',marker= 's')

plt.title('VALIDATION ACCURACY : ANN vs cnn')
plt.xlabel('Epochs')
plt.ylabel('Accurcay')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.subplot(1,2,2)
plt.bar(['ann', 'cnn'], [ann_test_acc,cnn_test_acc], color= ['#888888', '#2583eb'])

plt.title('Final_test_accuracy')
plt.ylabel('Accuracy')


for i, v in enumerate([ann_test_acc, cnn_test_acc]):
    plt.text(i, v+0.002, f"{v*100:.2f}%", ha ='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'ann_vs_cnn_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()


print("step 8  visualising feature maps")
print("=" * 40)


feature_map_model = keras.Model(inputs=cnn_model.inputs, outputs = cnn_model.layers[0].output)

sample_image = X_test_cnn[0:1]

feature_maps = feature_map_model.predict(sample_image, verbose =0)

print(f"Feature maps shape: {feature_maps.shape}")

fig, axes = plt.subplots(1, 9, figsize =(15,2))
axes[0].imshow(X_test[0], cmap='gray')

for i in range(8):
    axes[i+1].imshow(feature_maps[0, :, : i])
    axes[i+1].set_title(f"Filter{i+1}", fontsize =9)
    axes[i+1].axis('off')
    
fig.suptitle("Original Image vs first 8 fetaute maps", fontsize = 11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_map.png'), dpi = 300, bbox_inches = 'tight')

plt.show()

print("step 9  test prediction on the new image")
print("=" * 40)

    
sample = X_test_cnn[1]

prediction = cnn_model.predict(sample.reshape(1,28,28,1), verbose = 0)

plt.figure(figsize=(4,4))
plt.imshow(X_test[1], cmap='gray')

print("\n STEP 10 : INTRERACTIVE CANVAS")
print("=" * 40)
print(" A window will open, will use mouse to draw the digits")
print("Click and drag the mouse to draw")
print(" PRESS SPACE to classify your drawn digit")
print( " press C to clear teh canvas")
print(" press Q to quit teh canvas")

canvas = np.zeros((280, 280), dtype = np.uint8)
drawing = False
last_x, last_y = None, None


def predict_drawn_digit(canvas_img):
    coords = cv2.findNonZero(canvas_img)
    x, y, w, h = cv2.boundingRect(coords)
    
    #crop to just the digit
    digit = canvas_img[y:y+h, x:x+w]
    
    size = max(w,h)
    pad = int(size * 0.2)

    padded = np.zeros((size + pad * 2, size + pad * 2), dtype = np.uint8)    
    
    x_offset = (padded.shape[1] - w) // 2
    y_offset = (padded.shape[0] - h) // 2
    
    padded[y_offset: y_offset+h, x_offset: x_offset+w] = digit


   # resize the 28 by 28 mnist
    img_small = cv2.resize(padded, (28,28), interpolation=cv2.INTER_AREA)
    
    img_small = cv2.GaussianBlur(img_small, (3,3), 0)
    
    img_norm = img_small.astype('float32')/ 255.0
    img_cnn = img_norm.reshape(1, 28, 28, 1)
    
    
    #predict
    
    probs = cnn_model.predict(img_cnn, verbose =0)[0]
    pred = np.argmax(probs)
    conf = probs[pred] * 100
    
    return pred, conf, probs


def mouse_draw(event, x, y, flags, param):
    global drawing, last_x, last_y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(canvas, (last_x, last_y), (x,y), 255, thickness = 25)
        last_x, last_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
cv2.namedWindow('Draw a Digit (SPACE=predict, C=clear, Q=quit)')
cv2.setMouseCallback('Draw a Digit (SPACE=predict, C=clear, Q=quit)', mouse_draw)

prediction_text = "DRAW A DIGIT, THEN PRESS SPACE"

while True:
    
    display = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2BGR)
    
    cv2.putText(display, prediction_text, (10,260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,0), 2)
    
    cv2.imshow('Draw a Digit (SPACE=predict, C=clear, Q=quit)', display)
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif key == ord('c'):
        canvas[:] = 0
        prediction_text = 'CANAVS CLEARED DRAW AGAIN'
    elif key == ord(' '):
        if canvas.max() > 0:
            pred, conf, probs = predict_drawn_digit(canvas)
            prediction_text = f"Prediction: {pred} ({conf: .0f}% confient)"
            print(f"\n Predicted : {pred} | Confidence: {conf: .1f}%")
        else:
            prediction_text = "CANVAS IS EMPTY ! DRAW FIRST"
            
cv2.destroyAllWindows()

