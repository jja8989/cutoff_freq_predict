# Cutoff Frequency Determination Algorithm for Ferroelectric Device Pulse Measurements

This repository contains the implementation and supplementary materials for the paper  
**"Cutoff Frequency Determination Algorithm for Ferroelectric Device Pulse Measurements with Application to Machine Learning-Based Prediction."**

---

## 📘 Overview

Ferroelectric (FE) devices such as **FeRAM** and **FeFET** are widely used due to their spontaneous polarization properties.  
Accurate measurement and noise filtering of FE pulse signals are essential for analyzing intrinsic device characteristics such as remnant polarization, coercive field, and charge density.

This work proposes a **deterministic algorithm** and a **deep learning model** for automated cutoff frequency selection and signal denoising in ferroelectric pulse measurements.

---

## 🚀 Key Contributions

1. **Deterministic Cutoff Frequency Algorithm**
   - Identifies plateau regions in the **log-MSE–frequency** curve between raw and denoised signals.  
   - Determines the cutoff frequency as the minimum point of the **inverse gradient** of the log-MSE.
   - Provides a reproducible and quantitative criterion for noise filtering.

2. **Deep Learning Model for Prediction**
   - A hybrid **1-D dilated CNN + GRU** model that predicts the cutoff frequency directly from raw signals.
   - Learns both **local waveform details** and **global temporal dependencies**.
   - Remains robust even under **low sampling resolution** or limited data conditions.

3. **Automated Post-Processing**
   - Calculates current and charge density from denoised voltage signals.
   - Extracts \( Q_{\mathrm{charge}} \), \( Q_{\mathrm{res}} \), and \( Q_{\mathrm{discharge}} \) for ferroelectric characterization.

---

## 🧠 Algorithm Description

The deterministic algorithm performs the following steps:

1. Apply FFT to convert the raw signal to the frequency domain.  
2. Compute the MSE between raw and low-pass-filtered signals across candidate frequencies.  
3. Take the logarithm of MSE and compute its gradient with respect to frequency.  
4. Invert the gradient to highlight plateau regions:  
   \[
   G(f) = \frac{df}{d(\log_{10} \mathrm{MSE}(f))}
   \]
5. Identify the frequency \( f_c = \arg\min_f G(f) \) and apply a 15 % margin:
   \[
   f_c^{\text{final}} = 1.15 \times f_c
   \]
6. Reconstruct the denoised signal via inverse FFT and correct DC offset.

---

## 🧩 Deep Learning Architecture

<p align="center">
  <img src="figures/model_structure.png" width="80%">
</p>

- **Input:** Raw voltage signal  
- **Layers:**  
  - Three residual 1-D dilated convolution blocks (dilations = 1, 2, 4)  
  - Bidirectional GRU (hidden dim = 64 × 2)  
  - Global Avg + Max Pooling  
  - MLP Regression Head  
- **Output:** Predicted log-cutoff frequency \( \log_{10}(f_c) \)  
- **Loss:** Mean-squared error between predicted and algorithm-labeled cutoff frequencies  

---

## 📊 Dataset

- **Device:** Hf₁₋ₓZrₓO₂-based MFM structure  
- **Samples:** 7,492 raw signals  
- **Augmentation:** Downsampled (½ and ⅓ points) for robustness  
- **Total Samples:** 22,476  
- **Split:** 70 % train | 15 % validation | 15 % test  

---

## ⚙️ Results

| Dataset | MAE | MSE | MAPE |
|----------|-----|-----|------|
| Full data | 0.062 | 0.0116 | 15.81 % |
| Half sampling | 0.059 | 0.0114 | 15.14 % |
| One-third sampling | 0.063 | 0.0121 | 15.77 % |

- The model accurately reproduces the algorithm’s cutoff estimation.  
- Maintains performance even with significantly reduced sampling rates.  
- Enables **automated, consistent**, and **scalable** ferroelectric signal processing.

---