# Interoperability of Machine Learning Models in Earth Observation

This repository contains the code, trained models, export pipelines, and evaluation results 
for the Master's thesis:

**"Interoperability of Machine Learning Models in Earth Observation:
A Framework and Version Comparison between TensorFlow and PyTorch"**

---

## Overview

This work focuses on reproducibility and cross-framework portability of deep learning models 
(ConvLSTM) for Earth Observation.  
Three implementations are provided:

1. **Keras (TensorFlow backend)** – `models/keras_tf`
2. **Keras (PyTorch backend)** – `models/keras_pt`
3. **Native PyTorch** – `models/pytorch`

These models were trained on radar-based rainfall nowcasting data and used to study:
- Interface-level differences between Keras and PyTorch
- Backend-level differences (TensorFlow vs. PyTorch)
- Cross-framework interoperability using **ONNX** and **Keras 3**

The Keras-based model structure as well as the general modeling approach 
are inspired by the study of Petros Demetrakopoulos (2023).  
Data and the original model code for rainfall nowcasting can be found here:  
[https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast](https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast)

---

## Repository structure

```
models/            # Trained models (.keras, .pt) with STAC-ML metadata
exports/           # Converted models (ONNX, TF<->PT conversions)
training/          # Training & validation scripts for each framework
notebooks/         # Jupyter notebooks for evaluation and visualization
docs/              # Extended documentation, results and figures
references/        # Links to external datasets and publications
```

---

## Data

The training and validation data used in this thesis are **not included here**.

Data access and preprocessing instructions are available in:
```references/DATASETS.md```


The original dataset and code used as a basis for this work come from:  
[https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast](https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast)

---

## How to reproduce

1. Clone the repository  
2. See the corresponding `training/` folder for each model type
3. Use the same framework version as described in the thesis:
   - TensorFlow 2.19
   - PyTorch 2.7
4. STAC-ML metadata are stored alongside each model for documentation.

---

## License

MIT License.  
See [LICENSE](LICENSE).

---

## Citation

If you use these models or code, please cite:
```
Giese, B. (2025).
Interoperability of Machine Learning Models in Earth Observation:
A Framework and Version Comparison between TensorFlow and PyTorch.
Master’s Thesis, University of Münster.
```

