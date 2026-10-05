import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
from ucimlrepo import fetch_ucirepo

def main():
    print("Fetching dataset...")
    try:
        # Fetch dataset from UCI
        dataset = fetch_ucirepo(id=697)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        print("Dataset loaded successfully from UCI!")
    except Exception as e:
        print(f"Failed to load from UCI: {e}")
        print("Generating synthetic data as fallback...")
        from sklearn.datasets import make_classification
        # Generate synthetic data
        X_synth, y_synth = make_classification(n_samples=4000, n_features=10, n_informative=5, random_state=42)
        # Create a mock dataframe with similar columns
        columns = [
            'Curricular units 1st sem (grade)', 
            'Curricular units 2nd sem (grade)', 
            'Attendance', # mapped to something similar
            'Tuition fees up to date', 
            'Scholarship holder', 
            'Age at enrollment',
            'Course',
            'Marital status',
            'Previous qualification (grade)',
            'Admission grade'
        ]
        X = pd.DataFrame(X_synth, columns=columns)
        
        # Adjust some columns to be categorical/boolean like
        X['Tuition fees up to date'] = (X['Tuition fees up to date'] > 0).astype(int)
        X['Scholarship holder'] = (X['Scholarship holder'] > 0).astype(int)
        X['Course'] = np.random.randint(1, 10, X.shape[0])
        X['Marital status'] = np.random.randint(1, 5, X.shape[0])
        X['Age at enrollment'] = np.random.randint(18, 45, X.shape[0])
        
        y = pd.DataFrame(y_synth, columns=['Target'])
        y['Target'] = y['Target'].map({0: 'Dropout', 1: 'Graduate'})
        df = pd.concat([X, y], axis=1)

    print("\nDataset Shape:", df.shape)
    
    # Target variable handling
    # The UCI dataset Target has 'Dropout', 'Enrolled', 'Graduate'
    # We want to predict Dropout vs Non-Dropout (or just Dropouts vs Graduates)
    # The assignment is "identify students who are at risk of discontinuing their course"
    # Let's filter out 'Enrolled' for binary classification (Dropout vs Graduate)
    if 'Target' in df.columns:
        df = df[df['Target'] != 'Enrolled']
        df['Target'] = df['Target'].map({'Dropout': 1, 'Graduate': 0})
        y = df['Target']
        X = df.drop('Target', axis=1)
        
    print("\nClass distribution (1 = Dropout, 0 = Graduate):")
    print(y.value_counts())

    # Create directory for EDA plots
    os.makedirs("eda_plots", exist_ok=True)

    # EDA
    print("\nPerforming EDA and saving plots...")
    # Relationship between fee payment status and dropout
    if 'Tuition fees up to date' in df.columns:
        plt.figure(figsize=(8,6))
        sns.countplot(data=df, x='Tuition fees up to date', hue='Target')
        plt.title('Dropout by Tuition Fees Up to Date (1=Yes, 0=No)')
        plt.savefig('eda_plots/fee_vs_dropout.png')
        plt.close()

    # Preprocessing
    print("\nHandling missing values and encoding...")
    # Fill missing values with median for numeric, mode for categorical (if any)
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col].fillna(X[col].mode()[0], inplace=True)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        else:
            X[col].fillna(X[col].median(), inplace=True)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(list(X.columns), 'feature_names.pkl')

    # Models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    # k-fold cross-validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_model_name = ""
    best_recall = -1
    best_model = None

    print("\nTraining models and evaluating...")
    results = []
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        # Cross validation for Recall
        cv_recall = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='recall')
        print(f"CV Recall (Mean): {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
        
        # Train on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_scaled)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1
        })
        
        print(f"Test Accuracy:  {acc:.4f}")
        print(f"Test Precision: {prec:.4f}")
        print(f"Test Recall:    {rec:.4f}")
        print(f"Test F1-score:  {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        # Select best model based on Recall (as per case study requirement)
        if rec > best_recall:
            best_recall = rec
            best_model_name = name
            best_model = model

    print("\n--- Comparative Study ---")
    results_df = pd.DataFrame(results).sort_values(by='Recall', ascending=False)
    print(results_df.to_string(index=False))

    print(f"\nBest model based on Recall: {best_model_name} (Recall = {best_recall:.4f})")
    
    # Save best model
    joblib.dump(best_model, 'best_model.pkl')
    print("Best model and scaler saved successfully for deployment.")
    
    # Final Analysis Q&A
    print("\n--- Feature Importances ---")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df.head(10).to_string(index=False))
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        print(feature_importance_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
