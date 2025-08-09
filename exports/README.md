
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
- `keras_to_onnx.py` – Converts Keras or PyTorch models to ONNX (only works for pt backend)
- `export_to_tf.py` – Converts ONNX to TensorFlow SavedModel
- `export_to_pt.py` – Converts ONNX to PyTorch

---

## Notes

- All exports are intended for **reproducibility tests** in the thesis.  
- Accuracy and layer-by-layer equivalence may vary depending on export path and operator support.  
- For exact export commands and parameters, refer to `export_scripts/` and thesis methodology section.

---
