import pandas as pd

def load_and_preprocess_data(df):
    # Ensure df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data is not a DataFrame")

    # Check for required columns before processing
    required_columns = ['is_swing', 'plate_location_x', 'plate_location_z']
    if not all(column in df.columns for column in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Preprocess the already loaded DataFrame
    df = df[df['is_swing'] != 1]
    df = df.dropna(subset=['plate_location_x', 'plate_location_z'])

    # Fill missing values in 'spin_rate' with the column's mean
    if 'spin_rate' in df.columns:
        df['spin_rate'] = df['spin_rate'].fillna(df['spin_rate'].mean())

    # Convert categorical features to dummy variables
    df = pd.get_dummies(df, columns=['pitch_type', 'inning'], drop_first=True)
    
    print("Data preprocessing completed.")
    return df



