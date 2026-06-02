# Digital Image Processing Environment Setup Guide

## Windows Installation

### Step 1: Download Anaconda

Download Anaconda from the official website:

https://www.anaconda.com/download

### Step 2: Install Anaconda

Run the installer and follow these steps:

* Next
* I Agree
* Just Me
* Default Installation Path
* Install
* Wait for installation to finish
* Finish

### Step 3: Open Anaconda Prompt

1. Press **Windows Key**
2. Search **Anaconda Prompt**
3. Open it

You should see:

```bash
(base) C:\Users\Student>
```

### Step 4: Verify Installation

```bash
python --version
conda --version
```

### Step 5: Create Environment

Create a Digital Image Processing environment:

```bash
conda create -n dip_env python=3.11 -y
```

### Step 6: Activate Environment

```bash
conda activate dip_env
```

Prompt changes from:

```bash
(base)
```

to:

```bash
(dip_env)
```

This means the environment is active.

### Step 7: Install Required Libraries

```bash
pip install numpy matplotlib opencv-python
```

### Step 8: Install Spyder

```bash
conda install spyder -y
```

Launch Spyder:

```bash
spyder
```

---

# macOS Installation

## Step 1: Download Anaconda

Download Anaconda for macOS:

https://www.anaconda.com/download

Choose the installer based on your Mac:

* Apple Silicon (M1/M2/M3/M4)
* Intel Processor

## Step 2: Install Anaconda

1. Open the downloaded `.pkg` file
2. Click Continue
3. Agree to the license
4. Install
5. Enter your password if prompted
6. Complete installation

## Step 3: Open Terminal

Press:

```text
Command + Space
```

Search for:

```text
Terminal
```

Open Terminal.

## Step 4: Verify Installation

```bash
python --version
conda --version
```

If `conda` is not recognized, restart Terminal and try again.

## Step 5: Create Environment

```bash
conda create -n dip_env python=3.11 -y
```

## Step 6: Activate Environment

```bash
conda activate dip_env
```

You should see:

```bash
(dip_env)
```

## Step 7: Install Required Libraries

```bash
pip install numpy matplotlib opencv-python
```

## Step 8: Install Spyder

```bash
conda install spyder -y
```

Launch Spyder:

```bash
spyder
```

---

# Verify Installation

Run Python:

```bash
python
```

Then execute:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2

print("NumPy Version:", np.__version__)
print("OpenCV Version:", cv2.__version__)
```

If no errors appear, the setup is successful.

---

# Frequently Asked Questions (FAQs)

## What is Anaconda?

Anaconda is a Python distribution that includes:

* Python
* Conda package manager
* Scientific computing libraries
* Development tools

It simplifies installation and dependency management.

## Why create a separate environment?

Using environments helps:

* Avoid package conflicts
* Keep projects isolated
* Use different Python versions for different projects

## What does `conda activate dip_env` do?

It switches your terminal to the `dip_env` environment so that all installed packages and Python versions inside that environment are used.

## What is NumPy?

NumPy is used for:

* Arrays
* Matrix operations
* Mathematical computations

## What is Matplotlib?

Matplotlib is used for:

* Plotting graphs
* Displaying images
* Visualizing results

Example:

```python
plt.imshow(image)
plt.show()
```

## What is OpenCV?

OpenCV (Open Source Computer Vision Library) is used for:

* Reading images
* Processing images
* Computer vision applications
* Video analysis

Example:

```python
img = cv2.imread("image.jpg")
```

## What is Spyder?

Spyder is a beginner-friendly IDE commonly used in:

* Data Science
* Machine Learning
* Digital Image Processing

Features:

* Code Editor
* Variable Explorer
* Python Console
* Plot Viewer

## Conda command not found

### Windows

Open **Anaconda Prompt** instead of Command Prompt.

### macOS

Run:

```bash
conda init
```

Restart Terminal afterward.

## How do I see all environments?

```bash
conda env list
```

## How do I delete an environment?

```bash
conda remove --name dip_env --all
```

## How do I install additional packages?

Using pip:

```bash
pip install pandas
```

Using conda:

```bash
conda install pandas
```

## How do I check installed packages?

```bash
pip list
```

or

```bash
conda list
```

## How do I deactivate an environment?

```bash
conda deactivate
```

---

# Quick Setup Commands

```bash
conda create -n dip_env python=3.11 -y

conda activate dip_env

pip install numpy matplotlib opencv-python

conda install spyder -y

spyder
```

Happy Learning :)