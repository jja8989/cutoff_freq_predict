# Cutoff Frequency Determination Algorithm for Ferroelectric Device Pulse Measurements

This repository contains the implementation and supplementary materials for the paper  
**"Cutoff Frequency Determination Algorithm for Ferroelectric Device Pulse Measurements with Application to Machine Learning-Based Prediction."**

---

## Overview

Ferroelectric (FE) devices such as **FeRAM** and **FeFET** are widely used due to their spontaneous polarization properties.  
Accurate measurement and noise filtering of FE pulse signals are essential for analyzing intrinsic device characteristics such as remnant polarization, coercive field, and charge density.

This work proposes a **deterministic algorithm** and a **deep learning model** for automated cutoff frequency selection and signal denoising in ferroelectric pulse measurements.

---

## Key Contributions

1. **Deterministic Cutoff Frequency Algorithm**
   - Identifies plateau regions in the **log-MSE–frequency** curve between raw and denoised signals.  
   - Determines the cutoff frequency based on the minimum point of the **inverse gradient of the logarithmic MSE curve**.
   - Provides a **reproducible and quantitative criterion** for noise filtering.

2. **Deep Learning Model for Prediction**
   - A hybrid **1-D dilated CNN + GRU** model that predicts the cutoff frequency directly from raw signals.
   - Learns both **local waveform details** and **global temporal dependencies**.
   - Remains robust even under **low sampling resolution** or limited data conditions.

3. **Automated Post-Processing**
   - Calculates current and charge density from denoised voltage signals.
   - Extracts key charge metrics such as **Q_charge**, **Q_res**, and **Q_discharge** for ferroelectric characterization.

---

## Algorithm Description

The deterministic algorithm performs the following steps:

1. Converts the raw signal into the frequency domain using FFT.  
2. Computes the mean squared error (MSE) between the original and low-pass filtered signals across candidate frequencies.  
3. Applies a logarithmic scale to the MSE curve and analyzes its gradient with respect to log frequency.  
4. Inverts the gradient to highlight plateau regions, then identifies the point where this inverted gradient reaches its minimum — corresponding to the optimal cutoff frequency.
5. Multiplies the detected cutoff by a small safety margin (15%).
6. Reconstructs the denoised signal using inverse FFT and applies DC offset correction.

---

## Deep Learning Architecture

<p align="center">
  <img src="model_structure.png" width="80%">
</p>

- **Input:** Raw voltage signal  
- **Layers:** 
  - Three residual 1-D dilated convolution blocks (dilation rates: 1, 2, and 4)  
  - Bidirectional GRU layer (hidden size: 64 per direction)  
  - Global average and max pooling  
  - Fully connected regression head  
- **Output:** Predicted logarithmic cutoff frequency  
- **Loss function:** Mean squared error between logs of predicted and algorithm-labeled cutoff frequencies  

---

## Dataset

- **Device:** Hf₁₋ₓZrₓO₂-based MFM structure  
- **Samples:** 7,492 raw signals  
- **Data augmentation:** Downsampled to half and one-third of original resolution  
- **Total samples:** 22,476  
- **Split ratio:** 70% training / 15% validation / 15% test  

---

## Results

| Dataset              | MAE (log10 Hz) | MSE (log10 Hz) | MAPE (%) |
|----------------------|---------------|----------------|----------|
| Test-original            | 0.069         | 0.0128         | 18.37 %  |
| Test-half        | 0.061         | 0.0106         | 15.64 %  |
| Test-third   | 0.063         | 0.0122         | 17.38 %  |

- The model reproduces the algorithm’s cutoff estimation with high fidelity.  
- Maintains accuracy even when the input signals are heavily downsampled.  
- Enables **automated, consistent, and scalable** ferroelectric signal processing.

---