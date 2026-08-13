# AquaVision

AI-powered underwater image enhancement and marine object detection system.

## Overview

AquaVision is a computer vision project designed to improve the visual quality of underwater images and detect marine objects.

The system combines underwater image preprocessing, image enhancement, object detection, and image quality evaluation into a single pipeline.

## Features

- Underwater image preprocessing
- Image enhancement
- Marine object detection using YOLO
- Image quality evaluation
- End-to-end processing pipeline
- Streamlit-based interface

## Pipeline

Input Image
    ↓
Preprocessing
    ↓
Image Enhancement
    ↓
YOLO Object Detection
    ↓
Quality Evaluation
    ↓
Output

# Project structure 

AquaVision/
├── app.py
├── pipeline.py
├── preprocessing.py
├── enhancement.py
├── detection/
│   ├── __init__.py
│   ├── yolo_detector.py
│   └── test_yolo.py
├── metrics/
│   ├── quality_metrics.py
│   └── test_metrics.py
├── .gitignore
└── README.md

## Tech Stack

Python
OpenCV
NumPy
PyTorch
YOLO
Streamlit

## installation 

git clone https://github.com/Khushansh08/AquaVision.git
cd AquaVision

conda create -n aquavision python=3.10
conda activate aquavision

pip install -r requirements.txt

## run 

streamlit run app.py