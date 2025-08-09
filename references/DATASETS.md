# Dataset References

This project uses radar-based rainfall nowcasting data originally prepared by **Petros Demetrakopoulos (2023)** for the study:

> Demetrakopoulos, P. et al. (2023).  
> *LSTM radar precipitation forecast*  
> [GitHub Repository](https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast)

---

## Data Access

- **Original dataset and preprocessing code**:  
  [https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast](https://github.com/petrosDemetrakopoulos/LSTM-radar-precipitation-forecast)

- **Source**: KNMI (Royal Netherlands Meteorological Institute) radar composites

- **Data type**: HDF5 files containing 5-minute interval radar precipitation intensity maps.

- **Spatial resolution**:  
  Native: 1 km (KNMI radar grid)  
  Resized in this work to: **315 × 344 pixels**

- **Temporal coverage**:  
  Multiple years (see original repository for details).  
  This thesis uses 36 consecutive frames per sequence:  
  - First 18 frames: **input sequence**  
  - Last 18 frames: **forecast target**

---

## Data in this Repository

The raw and processed training/validation data are **not included** in this repository due to size limitations.

A dedicated data repository containing:
- training sequences
- validation sequences
- preprocessing scripts

is available here:

**➡ [External Data Repository – Link to be inserted]**

---

## License & Attribution

- The original dataset is publicly available via KNMI and subject to their data usage policies.
- If you use this dataset in your work, please credit:
  - KNMI for the radar data
  - Petros Demetrakopoulos for the preprocessed dataset and baseline model code.

