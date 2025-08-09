
---

## Export Details

### 1. ONNX exports (`onnx/`)
- Contains `.onnx` files converted from:
  - `keras_tf` model
  - `keras_pt` model
  - `pytorch` native model
- Conversion tools:  
  - `tf2onnx` for TensorFlow models  
  - `torch.onnx.export` for PyTorch models  

### 2. TensorFlow conversions (`tf_converted/`)
- `pytorch` native model converted to TensorFlow SavedModel
- Conversion path: **PyTorch → ONNX → TensorFlow**
- `keras_pt` model
- Conversion path: **Keras → ONNX → TensorFlow**
- Conversion path: **keras → TensorFlow**
- `keras_tf` model
- Conversion path: **keras → TensorFlow**

### 3. PyTorch conversions (`pt_converted/`)
- `keras_pt` model 
- Conversion path: **Keras → ONNX → PyTorch**
- stored as `.pt` file

---

## Export Scripts (`export_scripts/`)

Scripts used for conversions:
- `keras_to_onnx.py` – Converts Keras models to ONNX (only works for pt backend)
- `keras_to_tfSavedModel.py` – converts keras file into tensorflow saved model
- `savedModel_to_onnx.txt` – comand to convert a tensorflow saved model into an ONNX model
- `onnx_to_pt.txt` – Command for converting an ONNX model into .pt model
- `pt_to_onnx.py` – Converts a pytorch model to onnx
- `onnx_to_tfSavedModel.txt` – Converts ONNX to TensorFlow SavedModel

---

## Notes

- All exports are intended for **reproducibility tests** in the thesis.  
- Accuracy and layer-by-layer equivalence may vary depending on export path and operator support.  
- For exact export commands and parameters, refer to `export_scripts/` and thesis methodology section.

---
