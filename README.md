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

**Activation Distribution**  
<img width="1601" height="374" alt="download" src="https://github.com/user-attachments/assets/93532ce5-6aa3-457f-9c90-c60417819679" />

**Gradient Distribution**
<img width="1606" height="374" alt="download" src="https://github.com/user-attachments/assets/706d97aa-4e4c-4e18-9bee-0cbab6197e45" />

**Weights Gradient Distribution**
<img width="1606" height="374" alt="download" src="https://github.com/user-attachments/assets/404a26ca-a747-436c-8e20-dac387012bcd" />


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
 
