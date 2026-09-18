import pandas as pd
from config import MODEL_FEATURES, GRADIENT_BOOSTING_PARAMS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from sklearn.impute import SimpleImputer

def train_gradient_boosting_model(df, features=MODEL_FEATURES, target='is_strike'):
    print("Columns available for training:", df.columns)
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        raise KeyError(f"The following features are missing and needed for training: {missing_features}")

    X = df[features]
    y = df[target]
    
    # Check for NaN values and handle them
    if X.isna().any().any():
        print("NaN values detected in features. Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
    else:
        X = X.values  # Convert to NumPy array for compatibility with pipeline

    # Initialize the Gradient Boosting model using parameters from the config file
    model = GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS)

    # Create a pipeline with scaling and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Train the model on the entire dataset
    pipeline.fit(X, y)

    # Predict probabilities and calculate the Brier score
    y_pred_prob = pipeline.predict_proba(X)[:, 1]
    brier_score = brier_score_loss(y, y_pred_prob)

    # Append predictions to the DataFrame
    df['predicted_prob_strike'] = y_pred_prob

    print(f"Model training completed. Overall Brier Score: {brier_score:.4f}")
    
    return df, brier_score
