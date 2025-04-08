

# SLCA Model: Simulating Human Visual Perception and Eye-Tracking

## 📘 Overview

This project explores the **Spatial Leaky Competing Accumulator (SLCA)** model, an extension of Usher and McClelland’s 2001 Leaky Competing Accumulator framework. The SLCA integrates key cognitive functions—like information leakage, recurrent self-excitation, nonlinear dynamics, and random noise—to simulate human gaze behavior and perceptual decision-making.

## 🎯 Objectives

- Recreate brain activity during visual tasks using computational models.
- Simulate **eye-tracking behavior** using SLCA dynamics.
- Analyze and visualize the impact of spatial attention mechanisms on gaze patterns.

## 🧠 Background

The SLCA model combines neuroscientific insights with computational modeling to understand **visual perception**. It is designed to account for:

- **Leaky accumulation** of visual input
- **Self-reinforcing attention** toward specific spatial regions
- **Stochasticity** in gaze patterns
- **Non-linear decision thresholds**

These features allow SLCA to mimic observed human eye-tracking data with high fidelity.

## 🧰 Features

- Visual input grid to simulate spatial attention
- Adjustable parameters for leak rate, excitation, and noise
- Realistic simulation of human gaze shifts
- Graphs and animations for visualizing results

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/slca-vision-model.git
cd slca-vision-model
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python simulate_slca.py
```


## 📂 Folder Structure

```
📁 slca-vision-model/
│
├── simulate_slca.py         # Main simulation script
├── models/                  # SLCA model code
├── plots/                   # Visual output
├── data/                    # Input configuration (optional)
├── README.md
└── requirements.txt
```

## 📚 References

- Usher & McClelland (2001). The Leaky Competing Accumulator model.
- Zemliak (2022). Advances in computational models of visual perception.

## ✨ Future Work

- Integrate with real-time eye-tracking datasets
- Add GUI for interactive simulation control
- Compare SLCA with other perceptual models

---
