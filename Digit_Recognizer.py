#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:52:18 2026

@author: akshitmudgal
"""

#HANDWRIITEN DIGIT RECOGNIZER
#ANN, OPENCV, MATPLOTLIB, SEABORN


import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow import keras


print("=" * 65)
print("    HANDWRIITEN DIGIT RECOGNIZER - ANN MINI PROJECT")
print("=" * 65)

#STEP1: LOAD AND EXPLORE THE DATA

print("\n STEP1 : LAODING MNIST DATASET...")
print("=" * 65)

(X_train, y_train), (X_test, y_test)= keras.datasets.mnist.load_data()

print(f"Training images: {X_train.shape} --> {X_train.shape[0]} images, each {X_train.shape[1]} * {X_train.shape[2]} pixels")

print(f"Trainng Labels: {y_train.shape} --> {y_train.shape[0]} numbers (0-9)")

print(f"Test images : {X_test.shape}")

print(f"Pixel Range : [{X_train.min()}, [{X_train.max()}]")


fig, axes = plt.subplots(1, 10, figsize= (15,2))
fig.suptitle("MNIST SAMPLE HANDWRITTEN DATASET", fontsize = 12, fontweight = 'bold')

for digit in range(10):
    
    idx = np.where(y_train == digit)[0][0]
    axes[digit].imshow(X_train[idx], cmap ='gray')
    axes[digit].set_title(f"Label: {digit}", fontsize = 9, fontweight= 'bold')
    axes[digit].axis('off')
    
plt.tight_layout()

plt.savefig('project_step1_dataset.png', dpi=150, bbox_inches = 'tight')

plt.show()
print("DATSET VISULASIATION SAVED")


#STEP2: PREPROCESSING
print("\n STEP2 : preprocessing images...")
print("=" * 40)

sample_img = X_train[0]
print(f"Original SHAPE: {sample_img.shape}  <= 28 by 28 ")
sample_flat = sample_img.flatten()
print(f"After flatten {sample_flat.shape} <= 784 numbers in  a row ")

#(60000, 28 ,28) -> (60000, 784)

X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)


#normalize
X_train_norm = X_train_flat / 255.0
X_test_norm = X_test_flat / 255.0

X_train_final = X_train_norm[: 50000]
y_train_final = y_train [:50000]
X_val = X_train_norm[50000:]
y_val = y_train[50000:]

print("preprocessing completed..")
print(f" Train: {X_train_final.shape}, Validation: {X_val.shape}, Test: {X_test_norm.shape}")

#step3: ANN
print("\n STEP3 : BUILDING THE ANN...")
print("=" * 40)


model = keras.Sequential([
    #INPUT -> HIDDEN LAYER 1
    keras.layers.Dense(256, activation='relu', input_shape=(784,), name = 'hidden_1'),
    keras.layers.BatchNormalization(name="batchnorm_1"),
    keras.layers.Dropout(0.3, name='dropout_1' ),
    
    
    #HIDDEN LAYER 1  -> HIDDEN LAYER 2
    keras.layers.Dense(128, activation='relu', name = 'hidden_2'),
    keras.layers.BatchNormalization(name="batchnorm_2"),
    keras.layers.Dropout(0.3, name='dropout_2' ),
    
    #HIDDEN LAYER 2 --> HIDDEN LAYER 3
    keras.layers.Dense(64, activation='relu', name = 'hidden_3'),
    #keras.layers.BatchNormalization(name="batchnorm_3"),
    keras.layers.Dropout(0.3, name='dropout_3' ),
    
    #---OUTPUT LAYER---
    keras.layers.Dense(10, activation='softmax', name = 'output'),
    
    
    
    
    
    ])

model.summary()

total_params = model.count_params()
print(f"/n Total parameters the n/w will learns:{total_params}")


#step4: compile and train

print("\n STEP4 : compiling and training")
print("=" * 40)

model.compile(
    optimizer =keras.optimizers.Adam(learning_rate = 0.001),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
    
    )

early_stop = keras.callbacks.EarlyStopping(
    monitor ='val_accuracy',
    patience = 8,
    restore_best_weights = True,
    verbose = 1)

lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.5, #0.0005
    patience = 4,
    min_lr= 1e-6, # 1` * 10 ^-6 = 0.000001
    verbose = 1)

print("\n Training ...(~1-3 minutes depending upon ur hardware_ \n")

history = model.fit(
    
        X_train_final, y_train_final,
        epochs = 50,
        batch_size = 128,
        validation_data= (X_val, y_val),
        callbacks = [early_stop, lr_scheduler],
        verbose = 1
    )

print("\n Training is completed")

print("\n STEP 5 : Evaluationg the Model")
print("=" * 40)

fig, (ax1, ax2) = plt.subplots(1,2, figsize =(14,5))

ax1.plot(history.history['loss'], 'b-', linewidth = 2.5, label='Train Loss')
ax1.plot(history.history['val_loss'], 'r-', linewidth = 2.5, label='Val Loss')

ax1.set_xlabel('Epoch', fontsize = 12); ax1.set_ylabel('Loss',  fontsize=12)

ax1.set_title('training HIstory: Loss\n(Going Down = Learning!)', fontweight='bold')
ax1.legend(fontsize = 11)
ax1.grid(True, alpha = 0.3)

#right plot
ax2.plot(history.history['accuracy'], 'b-', linewidth = 2.5, label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], 'r-', linewidth = 2.5, label='Val Accuracy')

ax2.set_xlabel('Epoch', fontsize = 12); ax1.set_ylabel('Accuracy',  fontsize=12)

ax2.set_title('training HIstory: Accuracy\n(Going Up = Getting Better!)', fontweight='bold')
ax2.legend(fontsize = 11)
ax2.grid(True, alpha = 0.3)

best_epoch = np.argmax(history.history['val_accuracy'])
best_acc = max(history.history['val_accuracy'])


ax2.axvline(x = best_epoch, color='green', linestyle = '--', alpha = 0.7)


ax2.text(best_epoch + 0.5, best_acc - 0.03, f'Best epoch: {best_epoch+1}\n Acc: {best_acc: .2%}')

plt.suptitle('MNIST Digit Recognizer - Traning Progress', fontsize= 13, fontweight='bold')

plt.tight_layout()

plt.savefig('project_step5_traning.png', dpi =150, bbox_inches = 'tight')
plt.show()


test_loss, test_accuracy = model.evaluate(X_test_norm, y_test, verbose = 0)

print("=" * 40)
print(f"FINAL TEST ACCURCAY: {test_accuracy: .2%}")
print(f"FINAL TEST LOSS: {test_loss: .4f}")


#step 6 visulaise the predictions

print("\n STEP 6 : Visualising the predictions")
print("=" * 40)


y_pred_prob = model.predict(X_test_norm, verbose = 0) #(10000, 10)
#y_pred_prob[0] = [0.01, 0.02, 0.03...]

y_pred = np.argmax(y_pred_prob, axis = 1)

#row 0 : [0.01, 0.02, 0.01, 0.93]
#row 1 : [0.95, 0.01]

fig = plt.figure(figsize = (15, 8))

fig.suptitle("MODEL PREDICTION ON TEST DATA GREEN = CORRECT, RED = WRONG")

correct_idx = np.where(y_pred == y_test)[0][:12]
incorrect_idx = np.where(y_pred != y_test)[0][:4]

# 12 correct predictions

for i, idx in enumerate(correct_idx):
    ax = fig.add_subplot(3, 6, i+1)
    ax.imshow(X_test[idx], cmap = 'gray')
    confidence = y_pred_prob[idx, y_pred[idx]] * 100
    ax.set_title(f'Pred: {y_pred[idx]}\n ({confidence: .0f}% sure)', fontsize = 8, color='green', fontweight= 'bold')
    ax.axis('off')

#12 incorrect predictions

for i, idx in enumerate(incorrect_idx):
    ax = fig.add_subplot(3, 6, 13+i)
    ax.imshow(X_test[idx], cmap = 'RdPu')
    ax.set_title(f'True: {y_test[idx]}\n Pred: {y_pred[idx]} X', fontsize = 8, color='red', fontweight= 'bold')
    ax.axis('off')
    
plt.tight_layout()
plt.savefig('project_step6_predictions.png', dpi = 150, bbox_inches='tight')
plt.show()

print("\n STEP 6 : INTRERACTIVE CANVAS")
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
    img_flat = img_norm.reshape(1, 784)
    
    
    #predict
    
    probs = model.predict(img_flat, verbose =0)[0]
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


model.save("digit_recognizer_model.keras")
