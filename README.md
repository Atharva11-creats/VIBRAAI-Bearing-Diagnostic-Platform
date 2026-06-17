# VIBRAAI Bearing Diagnostic Platform 📊🔧

An AI-driven condition monitoring system designed for autonomous fault detection in rotary machinery components. This platform bridges classical mechanical vibration analysis with machine learning pipelines to evaluate bearing health metrics and predict structural failures in real time.

---

## 🚀 Key Features

* **Dual-Mode Portal:** Toggle seamlessly between analyzing historical operational baseline snapshots or inputting custom machinery vibration parameters.
* **Real-Time ML Inference:** Utilizes a trained Random Forest classifier to instantly categorize system status into three distinct operational zones: Normal Operation, Early Fault Warning, or Critical Failure Alarm.
* **Interactive Engineering Analytics:** Implements dynamic, responsive line charts for Time-Domain Waveforms and Fast Fourier Transform (FFT) Frequency Spectrums to visualize signal energy shifts.

---

## 🛠️ Tech Stack & Core Concepts

* **Programming Language:** Python
* **Web UI Framework:** Streamlit (Custom Dark Card Architecture Layout)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, Joblib serialization)
* **Data Processing & Visualization:** NumPy, Pandas, Plotly Express
* **Mechanical Domains:** Vibration Analysis (Root Mean Square (RMS), Kurtosis, Peak-to-Peak Amplitude, Skewness, Standard Deviation)