# main.py
from config import DATA_FILE_PATH, OUTPUT_FILE_PATH, OVERALL_STRIKE_RATE_COLUMN, INDIVIDUAL_ENTITIES, COMBINATION_ENTITIES, MODEL_FEATURES
from DataPreProcessing import load_and_preprocess_data
from IndividualSmoothRates import find_optimal_m, calculate_smoothed_rates
from CombinationsmoothRateCalculations import find_optimal_m_combination, calculate_combination_smoothed_rates
from Model import train_gradient_boosting_model
import pandas as pd

def main():
    
    # Load raw data without preprocessing
    df = pd.read_csv(DATA_FILE_PATH)
    print("Data loaded successfully.")
    
    # Calculate the overall strike rate
    overall_strike_rate = df[OVERALL_STRIKE_RATE_COLUMN].mean()
    print("Overall strike rate calculated:", overall_strike_rate)
    
    # Feature Engineering: Calculate individual smoothed rates
    optimal_m_individual = find_optimal_m(df, INDIVIDUAL_ENTITIES, overall_strike_rate)
    print("Optimal m for individual entities found:", optimal_m_individual)

    df = calculate_smoothed_rates(df, INDIVIDUAL_ENTITIES, overall_strike_rate, optimal_m_individual)
    print("Individual smoothed rates calculated and appended to DataFrame.")
    
    # Check that the new columns have been added for individual entities
    for entity, feature_name in INDIVIDUAL_ENTITIES.items():
        if feature_name in df.columns:
            print(f"Feature {feature_name} successfully added.")
        else:
            raise ValueError(f"Feature {feature_name} missing in DataFrame after individual smoothing.")

    # Feature Engineering: Calculate combination smoothed rates
    optimal_m_combination = find_optimal_m_combination(df, COMBINATION_ENTITIES, overall_strike_rate)
    print("Optimal m for combination entities found:", optimal_m_combination)

    df = calculate_combination_smoothed_rates(df, COMBINATION_ENTITIES, overall_strike_rate, optimal_m_combination)
    print("Combination smoothed rates calculated and appended to DataFrame.")
    
    # Check that the new columns have been added for combination entities
    for entity_pair, feature_name in COMBINATION_ENTITIES.items():
        if feature_name in df.columns:
            print(f"Feature {feature_name} successfully added.")
        else:
            raise ValueError(f"Feature {feature_name} missing in DataFrame after combination smoothing.")
    print("Columns after individual smoothed rates calculation:", df.columns)
    
    print("Sample of data after individual smoothed rates calculation:")
    print(df.head())

    print("Columns after combination smoothed rates calculation:", df.columns)
    print("Sample of data after combination smoothed rates calculation:")
    print(df.head())
    # Preprocess the data
    df = load_and_preprocess_data(df)
    print("Data preprocessed successfully.")
    
    # Train the model and save results
    df, brier_score = train_gradient_boosting_model(df, MODEL_FEATURES)
    print(f"Model training completed with Brier Score: {brier_score}")

    # Save the final DataFrame with predictions
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"Data with predictions saved to '{OUTPUT_FILE_PATH}'")

if __name__ == "__main__":
    main()
