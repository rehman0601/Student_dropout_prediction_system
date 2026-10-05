# Student Dropout Prediction System 🎓

## Overview
This Machine Learning project serves as an early warning system for educational institutions. It predicts whether a student is at high risk of dropping out based on their academic and socioeconomic profile. By identifying these students early, institutions can offer timely support and interventions.

## Project Structure
- `train_model.py`: The core machine learning script. It fetches data, performs preprocessing, trains multiple models (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting), and saves the best performing model.
- `app.py`: A **Streamlit** web application providing an interactive user interface to make real-time predictions using the trained model.
- `best_model.pkl`: The serialized version of the best performing machine learning model.
- `scaler.pkl`: The serialized `StandardScaler` used to normalize inputs.
- `feature_names.pkl`: The ordered list of features expected by the model.
- `eda_plots/`: Directory containing Exploratory Data Analysis visualizations.

## Setup & Installation

Ensure you have Python installed. It is recommended to use the provided virtual environment.

1. **Activate the virtual environment** (macOS/Linux):
   ```bash
   source venv/bin/activate
   ```
2. **Install required dependencies** (if not already installed):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit ucimlrepo joblib
   ```

## Usage

### 1. Training the Model
Before running the web app, you must train the model so it can generate the required `.pkl` files.
```bash
python train_model.py
```
This script will:
- Fetch the dataset from the UCI Machine Learning Repository (Dataset ID 697).
- Train and evaluate 5 different algorithms.
- Save the best model based on the **Recall** metric.

### 2. Running the Web App
Once the model is trained, you can launch the Streamlit dashboard:
```bash
streamlit run app.py
```
This will start a local server and open the app in your default web browser (usually `http://localhost:8501`). You can use the sidebar to load demo profiles ("High Risk" vs "Low Risk") or manually adjust the student attributes to see how they affect the dropout prediction.

## Why Recall?
The models are evaluated primarily on **Recall**. In a dropout prediction system, failing to identify a student who will drop out (a False Negative) is far worse than mistakenly flagging a good student as at-risk (a False Positive). High recall ensures that the maximum number of struggling students are correctly identified so they can receive help.
