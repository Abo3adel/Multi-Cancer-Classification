# Multi-Cancer Classification using DenseNet169
An end-to-end Deep Learning project that classifies 26 different subclasses of cancer from medical images. The model utilizes transfer learning with a pre-trained **DenseNet169** architecture, achieving an outstanding validation accuracy of **99%**.

## 📊 Dataset Overview
The dataset contains **130,002 images** categorized into **26 distinct classes** encompassing various types of cancer (e.g., Brain Tumor, Breast Malignant, Colon Adenocarcinoma, Lung Squamous Cell Carcinoma, etc.).
- **Input Size:** Resized to 224x224 pixels.
- **Normalization:** ImageNet standards `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.
- **Split:** 80% Training, 20% Validation.

## 🧠 Model & Training Strategy
The project leverages a **Two-Phase Fine-Tuning Strategy** to optimize the DenseNet169 model without destroying the pre-trained ImageNet weights:


- **Phase 1 (Classifier Training):** 
  - Frozen all feature extraction layers.
  - Trained only the fully connected classifier layer for 3 epochs.
  - *Optimizer:* Adam (lr = 1e-3).
- **Phase 2 (Deep Fine-Tuning):** 
  - Unfrozen the last dense block (`denseblock4`) and the final normalization layer (`norm5`).
  - Trained for an additional 2 epochs with a lower learning rate to capture domain-specific medical features.
  - *Optimizer:* Adam (lr = 1e-5 for features, 1e-4 for classifier).
 
## 📈 Results & Performance
The model demonstrated robust performance across all 26 classes, handling class variations effectively.
- **Final Validation Accuracy:** ~99.05%
- **Macro F1-Score:** 0.99
