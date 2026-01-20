<div align="center">

# 🚘 Vision-Based License Plate Recognition (ALPR)
### End-to-End Detection & OCR System

![Award](https://img.shields.io/badge/🏆_Award-Featured_Project-gold?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO-v11-FF0000?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

*A high-performance ALPR system featured on the University Projects Portal for its robust handling of real-world constraints.*

[View Code](alpr.py) • [Read the Report](Report.pdf)

</div>

---

## 🏆 Project Recognition
**This project was selected as a Featured Project by the University at Buffalo Department of Computer Science.** It was recognized for its comprehensive error analysis and the successful integration of deep learning detection with legacy OCR systems.

---

## 📖 Overview
Automatic License Plate Recognition (ALPR) in uncontrolled environments remains a challenge due to motion blur, oblique angles, and variable lighting. 

This project implements a two-stage pipeline:
1.  **Detection:** A custom-trained **YOLOv11** model to localize plates.
2.  **Recognition:** An image processing pipeline feeding into **Tesseract OCR** for text extraction.

We benchmarked state-of-the-art models (YOLOv11 vs. YOLOv12), finding that **YOLOv11** offered superior stability and accuracy for small, high-density targets like license plates.



## 🏗️ Architecture

```mermaid
graph LR
    A[Input Video] --> B[YOLOv11 Detection]
    B --> C{Plate Found?}
    C -- Yes --> D[Crop Region]
    D --> E[Adaptive Thresholding]
    E --> F[Tesseract OCR]
    F --> G[Text Output]
