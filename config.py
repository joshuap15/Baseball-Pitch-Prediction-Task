# config.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# File paths
DATA_FILE_PATH = "C:\\Users\\Admin\\Downloads\\pitch_data (2).csv"
OUTPUT_FILE_PATH = "C:\\Users\\Admin\\Downloads\\Side Projects\\NY Mets Prediction Task\\Repository\\FinalDataset\\final_data_with_predictions(3).csv"

# Hyperparameters and configurations
INDIVIDUAL_M_VALUES = [1, 5, 10, 20, 50, 100]
COMBINATION_M_VALUES = [1, 5, 10, 20, 50, 100]
GRADIENT_BOOSTING_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 3,
    'n_estimators': 300,
    'subsample': 1.0,
    'random_state': 42
}

# Feature Sets
MODEL_FEATURES = [
    'plate_location_x', 'plate_location_z', 'rel_speed', 'spin_rate', 'induced_vert_break', 'horizontal_break',
    'smoothed_pitcher_rate', 'smoothed_umpire_rate', 'smoothed_pitcher_umpire_rate', 'smoothed_catcher_umpire_rate', 'smoothed_batter_umpire_rate',
    'pitch_type_CH', 'pitch_type_FC','pitch_type_FF', 'pitch_type_FS', 'pitch_type_FT', 'pitch_type_OT','pitch_type_SL'
]

INDIVIDUAL_ENTITIES = {
    'pitcherid': 'smoothed_pitcher_rate',
    'hp_umpid': 'smoothed_umpire_rate',
    'cid': 'smoothed_catcher_rate',
    'batterid': 'smoothed_batter_rate'
}

COMBINATION_ENTITIES = {
    ('pitcherid', 'hp_umpid'): 'smoothed_pitcher_umpire_rate',
    ('cid', 'hp_umpid'): 'smoothed_catcher_umpire_rate',
    ('batterid', 'hp_umpid'): 'smoothed_batter_umpire_rate'
}

# Define as a placeholder if needed
OVERALL_STRIKE_RATE_COLUMN = 'is_strike'
