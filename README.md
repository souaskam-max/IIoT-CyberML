# IIoT-CyberML: A Multi-Dataset Detection System applied to OT networks in critical infrastructures

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)

## 📄 Abstract

This repository contains the source code and analysis framework for the research paper on **"IIoT-CyberML: A Multi-Dataset Detection System applied to OT networks in critical infrastructures, is an Unified Intrusion Detection for Industrial IoT using Hybrid Machine Learning Ensembles"**. We present a comprehensive analysis framework that harmonizes four major IIoT datasets (X-IIoTID, Edge-IIoTset, WUSTL-IIOT-2021, and TON-IoT) to train robust, cross-domain capable Intrusion Detection Systems.
IIoT-CyberML is a scalable machine learning framework for multi-class intrusion detection in Industrial IoT (IIoT) and OT environments. It fuses four major IIoT datasets (X-IIOTID, Edge-IIoTset, WUSTL-IIoT-2021, TON IoT) into a unified corpus and applies a robust preprocessing and ensemble learning pipeline. A soft voting ensemble achieves 99.84% accuracy, demonstrating strong cross-dataset generalization while highlighting challenges in rare attack detection. This project provides a solid foundation for securing critical infrastructures against evolving cyber threats.

## 📂 Repository Structure

```
.
├── data/               # Dataset placeholders and documentation
├── notebooks/          # Python code for analysis and training
├── results/            # Generated figures, tables, and logs
├── LICENSE            # MIT License
├── README.md          # Project documentation
└── requirements.txt   # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- GPU (Optional, but recommended for XGBoost training)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/iiot-ids-framework.git
   cd iiot-ids-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   Python Cyber_ML_IIOT.py
   ```

## 📊 Datasets

This framework utilizes the following datasets. Due to licensing and size constraints, the raw data is not included in this repository. Please download them from their respective sources and place them in the `data/` directory (or update the paths in the notebook).

- **X-IIoTID**: [Link to dataset]
- **Edge-IIoTset**: [Link to dataset]
- **WUSTL-IIOT-2021**: [Link to dataset]
- **TON-IoT**: [Link to dataset]

## 🛠️ Methodology

The analysis pipeline consists of:
1.  **Data Harmonization**: unifying feature spaces across disparate datasets.
2.  **Preprocessing**: Handling missing values, scaling, and encoding.
3.  **Model Training**: Training ensembles (XGBoost, Random Forest, etc.).
4.  **Evaluation**: Generating high-quality confusion matrices, ROC curves, and performance metrics.
5.  **Resource Analysis**: Monitoring inference latency and computational footprint for IIoT suitability.

## 📝 Citation

If you use this code in your research, please cite our work:

```bibtex
@article{SOUADIH2025IIoT,
  title={IIoT-CyberML: A Multi-Dataset Detection System applied to OT networks in critical infrastructures},
  author={Kamal SOUADIH and Foudil MIR},
  journal={.............................},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







