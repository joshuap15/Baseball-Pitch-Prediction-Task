# CombinationSmoothRateCalculations.py

# Imports from config and other necessary libraries
from config import COMBINATION_ENTITIES, COMBINATION_M_VALUES, OVERALL_STRIKE_RATE_COLUMN
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

def find_optimal_m_combination(df, combination_entities, overall_strike_rate, m_values=COMBINATION_M_VALUES):
    results = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    for m in m_values:
        print(f"Testing m value: {m}")
        log_loss_scores = []
        
        # Perform TimeSeriesSplit for cross-validation
        for train_index, test_index in tscv.split(df):
            try:
                train_data = df.iloc[train_index].copy()
                test_data = df.iloc[test_index].copy()
                
                # Initialize counts for smoothing calculations
                counts = {entity: {} for entity in combination_entities}
                strike_counts = {entity: {} for entity in combination_entities}
                smoothed_train_rates = {entity: [] for entity in combination_entities}
                smoothed_test_rates = {entity: [] for entity in combination_entities}

                # Calculate smoothed rates for training data
                for _, row in train_data.iterrows():
                    for entity_pair, feature_name in combination_entities.items():
                        pair_id = (row[entity_pair[0]], row[entity_pair[1]])
                        pair_strike_rate = strike_counts[entity_pair].get(pair_id, 0) / max(counts[entity_pair].get(pair_id, 1), 1)
                        smoothed_rate = (overall_strike_rate * m + pair_strike_rate * counts[entity_pair].get(pair_id, 0)) / (m + counts[entity_pair].get(pair_id, 0))
                        smoothed_train_rates[entity_pair].append(smoothed_rate)
                        counts[entity_pair][pair_id] = counts[entity_pair].get(pair_id, 0) + 1
                        strike_counts[entity_pair][pair_id] = strike_counts[entity_pair].get(pair_id, 0) + row['is_strike']

                # Add the calculated smoothed rates to the training DataFrame
                for entity_pair, feature_name in combination_entities.items():
                    train_data[feature_name] = smoothed_train_rates[entity_pair]

                # Recalculate smoothed rates for the test data
                for _, row in test_data.iterrows():
                    for entity_pair, feature_name in combination_entities.items():
                        pair_id = (row[entity_pair[0]], row[entity_pair[1]])
                        pair_strike_rate = strike_counts[entity_pair].get(pair_id, 0) / max(counts[entity_pair].get(pair_id, 1), 1)
                        smoothed_rate = (overall_strike_rate * m + pair_strike_rate * counts[entity_pair].get(pair_id, 0)) / (m + counts[entity_pair].get(pair_id, 0))
                        smoothed_test_rates[entity_pair].append(smoothed_rate)
                        
                # Add the calculated smoothed rates to the test DataFrame
                for entity_pair, feature_name in combination_entities.items():
                    test_data[feature_name] = smoothed_test_rates[entity_pair]

                # Fit the model and calculate log loss
                model = LogisticRegression()
                model.fit(train_data[list(combination_entities.values())], train_data['is_strike'])
                predictions = model.predict_proba(test_data[list(combination_entities.values())])[:, 1]
                score = log_loss(test_data['is_strike'], predictions)
                log_loss_scores.append(score)
                
            except KeyError as e:
                print(f"Key error for m={m}: {e}")
                break  # Stops processing this fold if there is an error
            
        # Calculate average log loss for this m value
        if log_loss_scores:
            average_log_loss = np.mean(log_loss_scores)
            results.append((m, average_log_loss))
            print(f"m={m}, Average Log Loss={average_log_loss}")
    
    # Return the optimal m value if results are available
    if results:
        optimal_m, _ = min(results, key=lambda x: x[1])
        return optimal_m
    else:
        print("Unable to determine optimal m due to issues during processing.")
        return None


def calculate_combination_smoothed_rates(df, combination_entities, overall_strike_rate, optimal_m):
    """
    Calculate smoothed rates for combined entities using the optimal m value.

    Args:
        df (pd.DataFrame): The dataset containing pitch data.
        combination_entities (dict): Dictionary with combined entities (like ('pitcherid', 'hp_umpid')) and their smoothed rate column names.
        overall_strike_rate (float): The overall strike rate in the dataset.
        optimal_m (int): Optimal m value for smoothing.

    Returns:
        pd.DataFrame: DataFrame with added smoothed rate columns for combined entities.
    """
    counts = {entity: {} for entity in combination_entities}
    strike_counts = {entity: {} for entity in combination_entities}
    smoothed_rates = {entity: [] for entity in combination_entities}

    # Calculate smoothed rates for each row in the dataframe
    for _, row in df.iterrows():
        for entity_pair, feature_name in combination_entities.items():
            pair_id = (row[entity_pair[0]], row[entity_pair[1]])
            pair_strike_rate = strike_counts[entity_pair].get(pair_id, 0) / max(counts[entity_pair].get(pair_id, 1), 1)
            smoothed_rate = (overall_strike_rate * optimal_m + pair_strike_rate * counts[entity_pair].get(pair_id, 0)) / (optimal_m + counts[entity_pair].get(pair_id, 0))
            smoothed_rates[entity_pair].append(smoothed_rate)
            counts[entity_pair][pair_id] = counts[entity_pair].get(pair_id, 0) + 1
            strike_counts[entity_pair][pair_id] = strike_counts[entity_pair].get(pair_id, 0) + row['is_strike']
    
    # Append smoothed rates as new columns in the dataframe
    for entity_pair, feature_name in combination_entities.items():
        df[feature_name] = smoothed_rates[entity_pair]
    
    return df

