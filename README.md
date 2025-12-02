# Signature Verification Using Siamese Network

## Overview
This project implements a **Siamese Neural Network** to verify handwritten signatures. The model learns to compare two signatures and determine whether they belong to the same person (**genuine**) or not (**forged**).

---

## What is a Siamese Network?

A **Siamese Network** is a type of neural network that learns **similarity between two inputs** instead of classifying them directly.  

- It consists of **two identical subnetworks (backbones)** that share the same weights.  
- Each backbone extracts an **embedding** (feature vector) from an input image.  
- A **distance function** (e.g., L1 or L2 distance) compares the embeddings.  
- The network is trained with **pairs of inputs** and a label indicating similarity (1 for similar, 0 for different).  

**How it works for signature verification:**

1. Two signature images are passed through the backbone CNN.
2. The network computes their embeddings.
3. The embeddings are compared using L1 distance.
4. A sigmoid layer outputs a probability:
   - High similarity → Genuine  
   - Low similarity → Forged  

**Diagram of Siamese Network:**  
![Siamese Network Diagram](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*hcj2L_XcDVlGhMDx.png)  


---

## Dataset
- Handwritten signature dataset with **train** and **test** sets.  
- Each sample is a pair of images with a label:  
  - `1` → Genuine  
  - `0` → Forged  

---

## Model Architecture
- **Backbone CNN:** Conv2D → MaxPooling → Dense → L2 Normalization  
- **Distance layer:** L1 distance between embeddings  
- **Output layer:** Sigmoid for binary classification  

---

## Results
| Metric      | Value |
|------------|-------|
| Training Accuracy | 92% |
| Test Accuracy     | 97% |

---

## Demonstration
🎥 **Video demo:** [Watch the demo video](https://drive.google.com/file/d/1jF7CpbFLoK_EU-9YZGGe-fC4YQfHmgS3/view)

---

## Future Improvements
- Use **pretrained backbones** (ResNet, EfficientNet) for better feature extraction.  
- Implement **data augmentation** to improve model robustness.  
- Deploy as a **web application** for online signature verification.  
- Add **interactive threshold slider** for controlling sensitivity.
