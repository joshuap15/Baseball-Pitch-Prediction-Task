# IndividualSmoothRates.py

# Imports from config and other necessary libraries
from config import INDIVIDUAL_ENTITIES, INDIVIDUAL_M_VALUES, OVERALL_STRIKE_RATE_COLUMN
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

def find_optimal_m(df, entities, overall_strike_rate, m_values=INDIVIDUAL_M_VALUES):
    results = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    for m in m_values:
        log_loss_scores = []
        
        for train_index, test_index in tscv.split(df):
            train_data = df.iloc[train_index].copy()
            test_data = df.iloc[test_index].copy()
            
            counts = {entity: {} for entity in entities}
            strike_counts = {entity: {} for entity in entities}
            smoothed_train_rates = {entity: [] for entity in entities}
            smoothed_test_rates = {entity: [] for entity in entities}

            # Calculate smoothed rates for the training set
            for _, row in train_data.iterrows():
                for entity, feature_name in entities.items():
                    entity_id = row[entity]
                    entity_strike_rate = strike_counts[entity].get(entity_id, 0) / max(counts[entity].get(entity_id, 1), 1)
                    smoothed_rate = (overall_strike_rate * m + entity_strike_rate * counts[entity].get(entity_id, 0)) / (m + counts[entity].get(entity_id, 0))
                    smoothed_train_rates[entity].append(smoothed_rate)
                    counts[entity][entity_id] = counts[entity].get(entity_id, 0) + 1
                    strike_counts[entity][entity_id] = strike_counts[entity].get(entity_id, 0) + row['is_strike']
                    
            # Apply the smoothed rates to train_data
            for entity, feature_name in entities.items():
                train_data[feature_name] = smoothed_train_rates[entity]

            # Calculate smoothed rates for the test set using training data counts
            for _, row in test_data.iterrows():
                for entity, feature_name in entities.items():
                    entity_id = row[entity]
                    entity_strike_rate = strike_counts[entity].get(entity_id, 0) / max(counts[entity].get(entity_id, 1), 1)
                    smoothed_rate = (overall_strike_rate * m + entity_strike_rate * counts[entity].get(entity_id, 0)) / (m + counts[entity].get(entity_id, 0))
                    smoothed_test_rates[entity].append(smoothed_rate)
                    
            # Apply the smoothed rates to test_data
            for entity, feature_name in entities.items():
                test_data[feature_name] = smoothed_test_rates[entity]

            # Fit logistic regression model and evaluate log loss on the test set
            model = LogisticRegression()
            model.fit(train_data[list(entities.values())], train_data['is_strike'])
            predictions = model.predict_proba(test_data[list(entities.values())])[:, 1]
            score = log_loss(test_data['is_strike'], predictions)
            log_loss_scores.append(score)
        
        average_log_loss = np.mean(log_loss_scores)
        results.append((m, average_log_loss))
        print(f"m={m}, Average Log Loss={average_log_loss}")
    
    optimal_m, _ = min(results, key=lambda x: x[1])
    return optimal_m


def calculate_smoothed_rates(df, entities, overall_strike_rate, optimal_m):
    counts = {entity: {} for entity in entities}
    strike_counts = {entity: {} for entity in entities}
    smoothed_rates = {entity: [] for entity in entities}

    for _, row in df.iterrows():
        for entity, feature_name in entities.items():
            entity_id = row[entity]
            entity_strike_rate = strike_counts[entity].get(entity_id, 0) / max(counts[entity].get(entity_id, 1), 1)
            smoothed_rate = (overall_strike_rate * optimal_m + entity_strike_rate * counts[entity].get(entity_id, 0)) / (optimal_m + counts[entity].get(entity_id, 0))
            smoothed_rates[entity].append(smoothed_rate)
            counts[entity][entity_id] = counts[entity].get(entity_id, 0) + 1
            strike_counts[entity][entity_id] = strike_counts[entity].get(entity_id, 0) + row['is_strike']
    
    for entity, feature_name in entities.items():
        df[feature_name] = smoothed_rates[entity]

    # Debug: Print column names to ensure they were added
    print("Columns after smoothed rates calculation:", df.columns)
    
    return df
