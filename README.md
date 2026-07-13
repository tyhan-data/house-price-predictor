# House Price Predictor 🏠

A machine learning application for predicting house prices using the Ames Housing Dataset with advanced feature engineering and the CatBoost algorithm.

**Live Application:** [https://house-price-predictor-by-tyhan.streamlit.app/](https://house-price-predictor-by-tyhan.streamlit.app/)

**Docker Hub:** [https://hub.docker.com/r/tyhan55/house-price-predictor](https://hub.docker.com/r/tyhan55/house-price-predictor)

---

## 📋 Project Overview

This repository contains a complete house price prediction system consisting of:

- **ML Model** - Created by [tyhan-data](https://github.com/tyhan-data)
- **Web Application** - Created by AI

The model is trained on the Ames Housing Dataset and achieves high prediction accuracy through careful feature engineering and hyperparameter optimization.

---

## 🎯 Features

- **Advanced Machine Learning Model**: CatBoost-based regression for accurate price predictions
- **Interactive Web App**: User-friendly Streamlit interface for making predictions
- **Comprehensive Data Analysis**: Jupyter notebook with exploratory data analysis and model development
- **Feature Engineering**: Optimized feature selection and preprocessing pipeline
- **Real-time Predictions**: Get instant house price estimates through the web application
- **Docker Support**: Easy deployment with containerized setup

---

## 📦 Repository Structure

```
house-price-predictor/
├── ames-house-price-prediction-with-catboost.ipynb  # ML Model Development & Analysis
├── app.py                                             # Streamlit Web Application
├── model/                                             # Trained Model Files
├── requirements.txt                                   # Python Dependencies
├── Dockerfile                                         # Docker Configuration
├── LICENSE                                            # MIT License
└── README.md                                          # This File
```

---

## 🛠️ Technology Stack

- **ML Framework**: CatBoost, Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Serialization**: Joblib
- **Containerization**: Docker

---

## 📋 Requirements

- Python 3.8+
- Docker & Docker Compose (for Docker setup)
- See `requirements.txt` for all dependencies

---

## ⚡ Installation & Usage

### Option 1: Local Setup

1. Clone the repository:
```bash
git clone https://github.com/tyhan-data/house-price-predictor.git
cd house-price-predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Streamlit application:
```bash
streamlit run app.py
```

5. Open your browser and navigate to `http://localhost:8501`

### Option 2: Docker Setup

#### Pull and Run from Docker Hub

1. Pull the pre-built Docker image:
```bash
docker pull tyhan55/house-price-predictor:latest
```

2. Run the container:
```bash
docker run -p 8501:8501 tyhan55/house-price-predictor:latest
```

3. Open your browser and navigate to `http://localhost:8501`

#### Build and Run Locally from Dockerfile

1. Clone the repository:
```bash
git clone https://github.com/tyhan-data/house-price-predictor.git
cd house-price-predictor
```

2. Build the Docker image:
```bash
docker build -t house-price-predictor:latest .
```

3. Run the container:
```bash
docker run -p 8501:8501 house-price-predictor:latest
```

4. Open your browser and navigate to `http://localhost:8501`

#### Docker Compose (Optional)

If you have a `docker-compose.yml` file in the repository:

```bash
docker compose up
```

Then access the application at `http://localhost:8501`

---

### Using the Web App

Visit the live application: [https://house-price-predictor-by-tyhan.streamlit.app/](https://house-price-predictor-by-tyhan.streamlit.app/)

---

## 📚 Model Development

The machine learning model was developed in the Jupyter notebook `ames-house-price-prediction-with-catboost.ipynb`, which includes:

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Training & Evaluation
- Hyperparameter Optimization
- Performance Metrics & Insights

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

- **Model Development**: [tyhan-data](https://github.com/tyhan-data)
- **Application Development**: AI Assistant
