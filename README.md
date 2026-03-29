# Skin Disease Detection Using Deep Learning

This project is a deep learning-based web application that classifies skin disease images into 7+ categories (Melanoma, Eczema, Psoriasis, Acne, Ringworm, etc.) using **Convolutional Neural Networks (CNN)** with **Transfer Learning**. It includes Grad-CAM visualizations to highlight regions contributing to the prediction.

---

## 🧩 Project Features

- Multi-class skin disease classification
- Uses **EfficientNetB3** as base model with fine-tuning
- Handles class imbalance with class weights
- Grad-CAM heatmaps for explainability
- Mobile-friendly web interface using **Streamlit**

---

## ⚡ Technologies & Tools

| Tool / Library        | Purpose |
|----------------------|---------|
| Python               | Core language |
| TensorFlow / Keras   | Model building and training |
| EfficientNet / DenseNet | Pre-trained CNN architectures |
| OpenCV / PIL         | Image loading & preprocessing |
| scikit-learn         | Class weights, evaluation metrics |
| Streamlit            | Web app deployment |

---

## 📁 Folder Structure

```text
Skin-Disease-Detection/
├── app.py                  # Streamlit UI
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── utils/
    └── gradcam.py          # Grad-CAM functions
    └── predict.py
    └── predict.py     
├── models/                 # Trained models (download separately)
└── dataset/                # Dataset (download separately)
