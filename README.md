# Oxford Flowers 102 Image Classifier

An end-to-end deep learning image classification project built with TensorFlow, Keras, and Python. This project consists of two core phases: an exploratory training notebook (`.ipynb`) utilizing Transfer Learning with TensorFlow Hub, and a production-ready Command Line Interface (CLI) utility (`predict.py`) designed for standalone inference.

This repository was completed as the final milestone project for the **Udacity AI Programming with Python Nanodegree**.

---

## 📁 Project Structure

```text
├── Project_Image_Classifier_Project.ipynb  # Training, validation, and visual evaluation notebook
├── predict.py                              # Standalone CLI prediction utility script
├── cat_to_name.json                        # Mapping file linking numeric IDs to real flower names
├── .gitignore                              # Prevents tracking local virtual environments (.venv)
└── README.md                               # Project documentation

## 3. Install Dependencies

pip install -r requirements.txt