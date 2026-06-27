# Transfer Learning & Advanced Vision --Study Notes
### (Topics Covered So Far)

---

## ✅ Covered: Transfer Learning, VGG16, ResNet, MobileNet
## ⏳ Pending: YOLO, Faster R-CNN, U-Net, Mask R-CNN

---

## 1. Transfer Learning --Core Idea

**Problem:** Naya CNN scratch se train karne ke liye lakhon images aur hafton ka time chahiye.

**Solution:** Ek pretrained model lo (jo already ImageNet --14 million images, 1000 categories --pe train ho chuka hai), aur uska knowledge "transfer" karo apne naye task mein.

**Kaise kaam karta hai:**
- Early layers edges/textures/shapes pehchaanti hain --yeh **general** hai, har image mein common hota hai
- Last layers task-specific decision leti hain
- Hum early layers ko **"freeze"** kar dete hain (weights lock, training ke time update nahi hote)
- Sirf last few layers replace/retrain karte hain apne task ke liye

**Fayda:** Kam data, kam time, achhi accuracy.

```python
from tensorflow.keras.applications import VGG16
base_model = VGG16(weights='imagenet', include_top=False)
base_model.trainable = False   # freeze
```

**Key terms:**
- **ImageNet** = 14 million+ images ka dataset, 1000+ categories --pretrained models isi pe train hote hain
- **Freeze** = layer ke weights ko lock karna, training ke time update na hone dena
- **include_top=False** = model ka sirf feature-extractor wala hissa lo, last classification layers nahi

---

## 2. VGG16 (2014, Oxford)

**Core Idea:** Sirf chhote 3x3 filters use karo, baar-baar stack karo (instead of bade filters).

**Insight:** Depth (zyada layers) badhane se accuracy badhti hai.

**Problem:** Bohot heavy --138 million parameters, ~528 MB size. Slow, mobile ke liye unfit.

---

## 3. ResNet (2015, Microsoft)

**Problem Jo Solve Ki --Degradation Problem:**
Jab CNN bohot deep (50-100+ layers) banaya gaya, accuracy **kharab** hone lagi (training data pe bhi) --sirf overfitting nahi, balki **vanishing gradient** ki wajah se: training ka "error signal" peeche jaate-jaate itna kamzor ho jaata hai ki shuru ki layers seekh hi nahi paatin.

**Solution --Skip Connection:**
Conv2D layers ko poora "exact answer" reproduce karne ko mat kaho --unhe sirf "kitna extra (residual) chahiye" seekhne do, aur original input ko ek **parallel shortcut wire** se direct add kar do final output mein.

```
Input (x) ────────────────────────────┐
   │                                    │
   └→ [Conv2D]→[ReLU]→[Conv2D] → F(x)   │
                                   │     │
                                   ↓     ↓
                        Final Output = F(x) + x
```

**Kyun Aasaan Hai:** Agar layer ka koi naya useful kaam nahi hai, usse sirf `F(x) ≈ 0` seekhna hai (jo weights ke natural zero-ke-paas starting point ke kareeb hai) --exact target reproduce karne se kahin aasaan. Isse layer "kam se kam harm nahi" karti → network 150+ layers tak deep ja sakta hai bina accuracy kharab kiye.

**Identity Mapping:** Jab layer ka best kaam ho "kuch mat badlo, input ko wahi output bana do" --kyunki feature already achha hai, zabardasti transform karne se bigad sakta hai.

**Versions:** ResNet18, 34, 50, 101, 152 (number = layers). ResNet50 sabse common transfer learning mein. ~25 million parameters --VGG se kam, accuracy zyada.

---

## 4. MobileNet (2017, Google)

**Problem:** Mobile/edge devices (phone, camera) pe real-time chalane ke liye VGG/ResNet bhi heavy hain.

**Core Idea --Depthwise Separable Convolution:**
Normal convolution ek hi step mein **spatial filtering + channel mixing** dono karta hai (expensive). MobileNet inhe **do alag, halke steps** mein todta hai:

1. **Depthwise Convolution** --har input channel ko apna alag 3x3 filter milta hai (sirf us channel pe, koi mixing nahi)
2. **Pointwise Convolution** --1x1 filter se filtered channels ko combine/mix karte hain

**Example (3 input, 4 output channels):**
- Normal Convolution: 108 multiplications (per pixel)
- MobileNet (Depthwise + Pointwise): 27 + 12 = 39 multiplications

Real layers mein (256/512 channels) yeh reduction **8-9x tak** hota hai.

**Trade-off:** Thodi accuracy ki keemat pe, dramatic speed/size improvement. ~4 million parameters, ~16 MB size.

---

## Comparison Table

| Feature | VGG16 | ResNet50 | MobileNet |
|---|---|---|---|
| MFD BY| 2014, Oxford | 2015, Microsoft | 2017, Google |
| Core Idea | 3x3 filters stack | Skip connections | Depthwise separable conv |
| Layers | 16 | 50 | ~28 |
| Parameters | 138 million | ~25 million | ~4 million |
| Model Size | ~528 MB | ~98 MB | ~16 MB |
| Speed | Slow | Medium | Fast |
| Best Use Case | Learning/research | Server/cloud, accuracy priority | Mobile/edge, speed priority |

---

## Note:

**VGG16 / ResNet / MobileNet** = Image Classification architectures (poori image ko ek label dete hain --"yeh cat hai"). Transfer learning mein backbone ki tarah use hote hain.

---
