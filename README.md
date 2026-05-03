# Deepfake Detection System using Vision Transformers

This project implements a **deepfake image detection system** using a **Vision Transformer (ViT)** with **attention-based token pruning** to improve efficiency while maintaining high accuracy.  
The system is designed for **fast and lightweight inference**, making it suitable for real-world and edge deployment scenarios.

---

## Project Overview

Deepfakes are increasingly realistic and pose serious challenges in media authenticity and security.  
Traditional CNN-based approaches focus on local features but may miss global inconsistencies.

This project leverages **Vision Transformers**, which use **self-attention** to capture global relationships across the image.  
To reduce computational cost, we introduce **token pruning**, where only the most important image patches are retained based on **CLS attention scores**.

---

## Key Features

- Vision Transformer–based deepfake detection  
- Attention-based **token pruning** for efficiency  
- ~59 percent token reduction  
- ~27 percent faster training  
- ~85 percent classification accuracy  
- CLS-attention mechanism for patch importance scoring  
- Lightweight and scalable architecture  

---

## Dataset

- FaceForensics++  
- Celeb-DF  

### Contains:
- Real images  
- Deepfake images generated using multiple techniques  

---

## System Architecture

1. **Input Image**
   - Resized to 224 by 224  
   - Converted into 16 by 16 patches  

2. **Patch Embedding**
   - Each patch converted into a 768-dimensional vector  
   - CLS token added for classification  

3. **Transformer Encoder**
   - Initial layers process all tokens  
   - CLS attention used to compute patch importance  

4. **Token Pruning**
   - Retain top 80 important patches  
   - Remove less relevant tokens  

5. **Classification Head**
   - Fully connected layers  
   - Output: fake or real  

---

## Technologies Used

- **Programming Language:** Python  
- **Deep Learning:** PyTorch  
- **Model:** Vision Transformer ViT-B/16  
- **Libraries:** Torchvision, NumPy, PIL  
- **Evaluation:** Scikit-learn  

---

## Evaluation Metrics

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

---

## Performance

- Accuracy: ~85 percent  
- Token reduction: ~59 percent  
- Training speed improvement: ~27 percent  

---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_vit.py
```

3. Run inference:

```bash
python predict_image.py
```

4. Provide image path when prompted

---

## Future Improvements

* Extend to video-level deepfake detection
* Improve robustness against unseen deepfake methods
* Optimize for mobile and edge deployment
* Integrate explainable AI for interpretability
