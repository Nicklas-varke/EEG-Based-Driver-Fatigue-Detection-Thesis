# Driver Fatigue Evaluation Using EEG-Based Spatio–Temporal CNN

This repository contains all source code and experiments developed as part of my Master's Thesis implementation based on the paper:

**Z. Gao et al., "EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation," IEEE Transactions on Neural Networks and Learning Systems.**

## 📘 Overview
The goal of this project is to detect and evaluate driver fatigue levels using EEG signals. The implementation reproduces the methodology from the referenced paper, focusing on spatial–temporal modeling of EEG channels via deep learning.

This branch contains **all scripts** 

## 🧠 Problem Description
Driver fatigue is a major factor contributing to road accidents. EEG signals provide a reliable physiological measure for identifying drowsiness. This work implements a deep learning pipeline that:
- Preprocesses raw EEG signals
- Learns spatio–temporal features using convolutional and temporal modules
- Classifies fatigue levels or performs regression on continuous fatigue metrics

The goal is to approximate or improve the performance reported by Gao et al.

## 🏗️ Architecture
The core network follows the **Spatio–Temporal Convolutional Neural Network (ST-CNN)** proposed in the paper.

- **Spatial CNN Layer:** Extracts spatial dependencies between EEG electrodes
- **Temporal CNN / 1D Conv Layers:** Captures temporal evolution
- **Feature Fusion Block:** Merges spatial and temporal representations
- **Fully Connected Layers:** For final classification

This code was developed as part of the Master's Thesis work, focusing on:
- Reproducing the results of the published paper
- Exploring generalization across subjects
- Evaluating the impact of spatial–temporal fusion layers
- Improving training stability and model explainability

## 📚 References
Z. Gao et al., "EEG-Based Spatio–Temporal Convolutional Neural Network for Driver Fatigue Evaluation," in IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 9, pp. 2755-2763, Sept. 2019, doi: 10.1109/TNNLS.2018.2886414. 

## 📝 License
This branch is for academic research purposes. Redistribution of datasets must follow their respective licenses.
