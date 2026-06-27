# 🩸 Automated WBC Detection & Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project provides an **end-to-end automated system** for detecting and classifying White Blood Cells (WBCs) from microscopic blood smear images. It replaces the tedious manual counting process with AI, delivering a detailed diagnostic report instantly.

- **Detection**: YOLOv8 localizes all WBCs in the image.
- **Classification**: EfficientNet-B0 classifies each WBC into one of four subtypes:
  - Neutrophil
  - Lymphocyte
  - Monocyte
  - Eosinophil
- **Web Application**: A professional web interface (Django + FastAPI) allows clinicians to upload images and receive a structured report (counts, N/L ratio, medical interpretation).

## Architecture

The project is built with a **decoupled microservice architecture** to ensure scalability and reliability:

1. **FastAPI Server** (Port `8001`): Loads the AI models (YOLO + EfficientNet) and exposes a `/predict` endpoint.
2. **Django Server** (Port `8000`): Hosts the web interface, handles user uploads, and communicates with the FastAPI backend.

## Project Structure

```text
blood/
├── api/                    # FastAPI inference server
│   └── main.py
├── blood_site/             # Django project settings
├── analyzer/               # Django app (views, uploads)
├── config/                 # YAML configuration files
├── src/                    # Core source code
│   ├── data/               # Dataset loaders & preprocessing
│   ├── models/             # YOLO & EfficientNet definitions
│   ├── training/           # Training loops & metrics
│   ├── inference/          # Full pipeline (detection + classification)
│   └── evaluation/         # Evaluation scripts
├── scripts/                # Executable scripts (training, evaluation)
├── notebooks/              # Jupyter notebooks (EDA, visualization)
├── templates/              # HTML templates for Django
├── outputs/                # (Ignored) Saved models and predictions
├── data/                   # (Ignored) Raw & processed datasets
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```
