# Pitch Prediction Analysis

This folder contains the complete workflow for a pitch prediction analysis, including data preprocessing, feature engineering, model development, and final predictions. The contents are organized into subfolders for easier navigation.

## Folder Structure

- **`data/`**: Contains the original dataset used for analysis.
- **`final_dataset/`**: Holds the final dataset with predictions appended.
- **`notebooks/`**: Includes Jupyter notebooks for Exploratory Data Analysis (EDA), Feature Engineering, and Model Development and Tuning.
- **`functions/`**: Contains the Python scripts needed to preprocess data, perform feature engineering, and run the model.

## Workflow Overview- Function of the Pipeline

1. **Data Preprocessing**: Cleans the raw data, handles missing values, and prepares it for modeling.
2. **Feature Engineering**: Calculates smoothed rates for individual and combination entities to create new features.
3. **Model Development**: Builds and tunes a Gradient Boosting model.
4. **Final Predictions**: Applies the trained model to the data and appends predictions to the dataset.

## How to Run the Code

### Prerequisites

- **Python 3.x** installed on your system
- Required Python libraries (install with `pip`):
    ```bash
    pip install pandas numpy scikit-learn
    ```

### Running the Code

1. **Navigate to the Folder**:
   - Open a terminal or command prompt.
   - Change to the directory where this project is located.

2. **Place the Original Dataset**:
   - Ensure the dataset is in the `data/` folder.

3. **Run the Pipeline**:
   - Change to the `functions/` directory:
     ```bash
     cd functions
     ```
   - Run the `main.py` script:
     ```bash
     python main.py
     ```
   - This will load and preprocess the data, perform feature engineering, and train the Gradient Boosting model. The final output will be saved as `final_data_with_predictions.csv` in the `final_dataset/` folder.

4. **Review the Results**:
   - Open the `final_dataset/` folder to find the CSV file containing the predictions.

### Additional Information

- **Scripts**: The Python scripts in `functions/` are modular, handling separate tasks like data loading, preprocessing, feature engineering, and model training.
- **Notebooks**: Jupyter notebooks in the `notebooks/` folder document the exploratory data analysis, feature engineering steps, and model tuning process.

## Author

- Joshua Poozhikala 
  
## License

This project is for educational purposes and is not distributed under a specific license.
