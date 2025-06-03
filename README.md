# 🚗 License Plate Recognition System (YOLOv11 + EasyOCR)

This project implements a complete pipeline for automatic **License Plate Recognition (LPR)** using deep learning. It combines object detection (YOLOv11) to locate plates and optical character recognition (EasyOCR) to decode the plate text.

---

## 🔧 Features

- **YOLOv11**-based number plate detection (fine-tuned on custom dataset)
- **EasyOCR** recognition module (custom-trained for license plate characters)
- Real-time video inference with FPS benchmark
- Exported models in **ONNX** and **TensorRT** formats
- Evaluation using precision, recall, F1-score, mAP, and Levenshtein distance
- Python-based, runs on CPU/GPU

---

## 📊 Dataset

- **Detection Dataset**: Annotated license plates in YOLOv11 format (from Roboflow)
- **Recognition Dataset**: Cropped plate images with label CSVs
- **Preprocessing**:
  - Resized to 640x640 for detection
  - Grayscale 64x600 for recognition
  - Train/Val/Test split: 70% / 20% / 10%

---

## 🧠 Models

### 1. Detection – YOLOv11

- Model: `yolo11s.pt` (Ultralytics)
- Fine-tuned for 100 epochs with early stopping
- Exported as `best_model_yolo11s.pt` and `onnx`

### 2. Recognition – EasyOCR

- Custom-trained with alphanumeric character set
- Character-level Accuracy: **53.81%**
- Word-level Accuracy: **33.33%**
- Average Levenshtein Distance: **3.71**

---

## 📈 Training Curves

YOLOv11 training includes:
- Loss vs. Epoch
- mAP@50 and mAP@50-95 curves

EasyOCR loss logged manually during training.

---

## ⚙️ Real-Time Performance (CPU vs GPU)

| Task                 | CPU (FPS) | GPU (FPS) |
|----------------------|-----------|-----------|
| Detection Only       |   1.47    |   11.18   |
| Detection + OCR      |   1.14    |   7.33    |

Tested on HP Laptop (i7, 8GB RAM, NVIDIA GeForce MX250 4 GB).

---

## 🧪 Evaluation

**Detection:**
- Precision, Recall, F1 Score
- mAP@50: 0.93, mAP@50-95: 0.68

**Recognition:**
- Character Accuracy, Word Accuracy
- Levenshtein Distance

---

## 📽️ Project Demo

Watch the full end-to-end License Plate Recognition system in action:

👉 [**Watch Demo on Google Drive**](https://drive.google.com/file/d/1iE0vXkGdoq38DrfTpZhSfNolg_jAUhMQ/view?usp=sharing)

> 🔹 Real-time detection and OCR  
> 🔹 Demo video: `demo.mp4`  
> 🔹 Includes FPS benchmark overlays



