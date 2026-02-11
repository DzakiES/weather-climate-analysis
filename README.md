# 🌦️ Weather Climate Data Analysis — BMKG IoT Smart System

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?logo=tensorflow)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-0078D4)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Project Overview

This project presents a **complete end-to-end time series analysis pipeline** on daily climate data from **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)** — Indonesia's National Weather Agency. The data covers weather observations from a geophysics station in **Bandung, West Java** throughout **2024**.

The project covers the full data science workflow:
> **Raw Data → Missing Value Imputation → Time Series Analysis → Decomposition → Feature Extraction → Anomaly Detection (ML & Deep Learning)**

This work was developed as part of the **Intelligent IoT Systems (CLO 2 & 3)** coursework at Telkom University.

---

## 📊 Dataset

- **Source**: [BMKG Data Online](https://dataonline.bmkg.go.id)
- **Period**: January 2024 – December 2024 (366 daily records)
- **Station**: Geofisika Bandung, West Java

**Weather Parameters Analyzed:**

| Parameter | Description | Unit |
|-----------|-------------|------|
| `TAVG` | Average Temperature | °C |
| `RH_AVG` | Average Relative Humidity | % |
| `RR` | Rainfall / Precipitation | mm |
| `FF_AVG` | Average Wind Speed | m/s |

---

## 🗂️ Project Structure

```
weather-climate-analysis/
│
├── data/
│   ├── laporan_iklim.csv                  # Raw BMKG climate data
│   └── laporan_iklim_harian-2024.xlsx     # Daily climate report (Excel)
│
├── Soal_1_Imputasi.ipynb                  # Missing value imputation
├── Soal_2_EDA_Stasioneritas.ipynb         # EDA & stationarity test
├── Soal_3_Dekomposisi.ipynb               # Time series decomposition
├── Soal_4_Ekstraksi_Fitur.ipynb           # Residual feature extraction
├── Soal_5_Anomaly_Detection_ML.ipynb      # Anomaly detection (RF & XGBoost)
├── Soal_6_Anomaly_Detection_LSTM.ipynb    # Anomaly detection (LSTM)
│
└── README.md
```

---

## 🔬 Notebooks Description

### 📓 Notebook 1 — Missing Value Imputation
Handles missing values coded as `8888` and `9999` in raw BMKG data using two methods:

- **Linear Interpolation** — fast, simple, suitable for gradual changes
- **Prophet** — Facebook's time series forecasting library, captures daily/weekly/yearly seasonality

**Evaluation**: Cross-validation with MAE and RMSE metrics. Prophet outperformed Linear Interpolation by accounting for seasonal patterns in weather data.

### 📓 Notebook 2 — EDA & Stationarity Test
Performs Exploratory Data Analysis and tests whether each weather parameter is stationary using:

- **Augmented Dickey-Fuller (ADF) Test**
- **ACF (Autocorrelation Function) plots**
- **PACF (Partial Autocorrelation Function) plots**

**Finding**: 3 out of 4 parameters (TAVG, RR, FF_AVG) are stationary; RH_AVG shows non-stationarity requiring differencing.

### 📓 Notebook 3 — Time Series Decomposition
Decomposes each parameter into its structural components using the **Additive Model**:

- **Trend** — long-term direction
- **Seasonal** — recurring periodic patterns
- **Residual** — remaining noise after removing trend and seasonality

**Key insight**: Trend correlation values |r| < 0.3 across all parameters confirm no significant linear trend — fluctuations are cyclical (seasonal), not directional.

### 📓 Notebook 4 — Residual Feature Extraction
Extracts residual components from decomposition results as features for anomaly detection. Includes:

- Residual time series plots for all 4 parameters
- Distribution analysis with ±2σ bounds
- Statistical summary of residual behavior per parameter

### 📓 Notebook 5 — Anomaly Detection with Machine Learning
Applies two regression-based anomaly detection models on residual data (TAVG & RR):

| Model | Anomalies Detected (TAVG) | Anomalies Detected (RR) |
|-------|--------------------------|------------------------|
| **Random Forest Regressor** | 2 | 9 |
| **XGBoost Regressor** | 1 | 4 |

Anomaly threshold: **±3σ** from prediction error distribution.

**Conclusion**: Random Forest is more sensitive (higher recall), while XGBoost is more precise (lower false positives). Choice depends on use case: early warning vs. precision detection.

### 📓 Notebook 6 — Anomaly Detection with Deep Learning (LSTM)
Implements a **multi-layer LSTM neural network** for sequence-based anomaly detection:

| Parameter | LSTM Architecture | Anomalies Detected |
|-----------|------------------|-------------------|
| TAVG | 64→32 LSTM + Dropout(0.2) | 0 |
| RR | 64→32 LSTM + Dropout(0.2) | 6 |

**LSTM Parameters:**
- Sequence length: 10 days
- Epochs: 50 (with early stopping, patience=10)
- Batch size: 16
- Threshold: ±3σ

**Sensitivity Ranking**: Random Forest (11) > LSTM (6) > XGBoost (5)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| **Pandas / NumPy** | Data manipulation & numerical computing |
| **Matplotlib** | Data visualization |
| **Prophet** | Time series forecasting & imputation |
| **Statsmodels** | ADF test, ACF/PACF, seasonal decomposition |
| **Scikit-Learn** | Random Forest, evaluation metrics |
| **XGBoost** | Gradient boosting regression |
| **TensorFlow / Keras** | LSTM deep learning model |

---

## 🚀 How to Run

1. **Clone this repository**
```bash
git clone https://github.com/DzakiES/weather-climate-analysis.git
cd weather-climate-analysis
```

2. **Install dependencies**
```bash
pip install pandas numpy matplotlib prophet statsmodels scikit-learn xgboost tensorflow openpyxl
```

3. **Run notebooks in order**
```bash
jupyter notebook
```

Execute notebooks sequentially:
`Soal_1` → `Soal_2` → `Soal_3` → `Soal_4` → `Soal_5` → `Soal_6`

> ⚠️ **Note**: Prophet and LSTM training may take several minutes depending on your hardware.

---

## 📈 Key Results Summary

| Analysis | Result |
|----------|--------|
| Missing value handling | Prophet imputation outperforms Linear Interpolation for seasonal weather data |
| Stationarity | 3/4 parameters stationary (ADF test) |
| Trend significance | No significant linear trend (|r| < 0.3 for all parameters) |
| Most sensitive anomaly detector | Random Forest Regressor (11 total anomalies) |
| Most precise anomaly detector | XGBoost Regressor (lowest MAE/RMSE) |
| Best for temporal patterns | LSTM (captures sequential dependencies) |

---

## 👤 Author

**Dzaki Endraghani Sunarko**
📧 dzakies2003@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/dzakisunarko/)
🎓 Informatics — Telkom University, Bandung

---

> *This project was developed as part of the Intelligent IoT Systems coursework (CLO 2 & 3) at Telkom University, 2025.*
