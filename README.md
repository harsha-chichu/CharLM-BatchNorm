# CharLM-MLP 🚀
*A character-level language model from scratch, building towards GPT-like architectures.*

## 📖 Overview
This project implements a **character-level language model** trained on a dataset of baby names. It follows the “Zero to Hero” progression (bigram → MLP → transformer), but all code is written **from scratch in PyTorch**, without using high-level modules like `nn.Linear` or `nn.Sequential`.

The goal is to deeply understand how embeddings, hidden layers, normalization, and training dynamics come together to generate text.

---

## 🔄 Evolution of the Model

### **1. Initial MLP Model**
- Context size = 3 characters.  
- Embeddings → hidden layer (tanh) → logits for next character.  
- Trained with cross-entropy loss.  
- Generated names, but training was unstable and loss curves were noisy.  

### **2. Improved Model with Batch Normalization**
- Added **manual BatchNorm** between the hidden pre-activations and `tanh`.  
- Purpose: stabilize distributions, avoid saturation, improve gradient flow.  
- Implemented running mean/std for inference mode.  
- Result:  
  - Faster convergence.  
  - Smoother loss curves.  
  - More coherent sampled names.  

---

## 📊 Results

**Training Loss Curve:**  
*![Activation Distribution](<img width="1601" height="374" alt="image" src="https://github.com/user-attachments/assets/47871885-3bf6-4152-8bac-f1bb782b5dbe" />)*  

*![Gradient Distribution](<img width="1606" height="374" alt="image" src="https://github.com/user-attachments/assets/c6d75c13-0b59-4579-bae9-475d18eebd41" />)*

*![Weights Gradient Distribution](<img width="1606" height="374" alt="image" src="https://github.com/user-attachments/assets/25fdc73c-b82f-4b01-bd6b-72bcebc11dfc" />)*

**Generated Sample Names:**  
- `Loma`  
- `Deren`  
- `Mariel`  
- …  

---

## ⚙️ How It Works
1. **Data prep**:  
   - Reads `names.txt`.  
   - Builds vocabulary + integer mappings.  
   - Creates `(context, next_char)` training pairs.  

2. **Model**:  
   - Embedding matrix for characters.  
   - Flattened embedding → linear layer → **BatchNorm → tanh** → output layer.  
   - Cross-entropy loss.  

3. **Training loop**:  
   - Manual forward/backward passes.  
   - Mini-batch SGD.  
   - Plots loss for training/dev sets.  

4. **Sampling**:  
   - Generates new names one char at a time until `.`.  

---

## 🧠 Key Learnings
- How embeddings compress character information into dense vectors.  
- Why **BatchNorm** stabilizes training and reduces internal covariate shift.  
- How manual backpropagation deepens understanding of PyTorch internals.  
- The importance of architectural tweaks for training stability.  

---

## 🚀 Next Steps
- Add **Dropout** for regularization.  
- Train on a **larger, real-world dataset**.  
- Extend to **multi-layer MLP** or a **Transformer block**.  
- Compare **with vs without BatchNorm** quantitatively.  

---

## 📂 Repo Structure
```
CharLM-MLP/
│── CharLM_MLP_2.ipynb   # Notebook with BatchNorm implementation
│── names.txt            # Baby names dataset
│── README.md            # Project documentation
```

---
 
