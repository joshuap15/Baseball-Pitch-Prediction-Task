## How to Run the Code

### 1. Update File Paths

Before running the pipeline, **open the `config.py` file** and update the following paths:

- **Raw Dataset Path**: Set the path where your raw dataset is located. For example:

    ```python
    RAW_DATA_PATH = "path/to/your/dataset.csv"
    ```

- **Output Path**: Set the path where you'd like the final dataset with predictions to be saved. For example:

    ```python
    OUTPUT_DATA_PATH = "path/to/your/output/final_data_with_predictions.csv"
    ```

Make sure these paths point to the correct locations on your system.

### 2. Run the Full Pipeline

Once you have updated the paths in `config.py`, you can run the pipeline as follows:

1. **Navigate to the `functions/` directory**:

    ```bash
    cd functions
    ```

2. **Run the main script**:

    ```bash
    python main.py
     ```
Also after making the changes you can run the main.py code in your environment and the full pipeline will be executed


This will trigger the following process:
- Load and preprocess the raw data using `DataPreProcessing.py`.
- Perform feature engineering by calculating individual and combination smoothed rates via `IndividualSmoothRates.py` and `CombinationsmoothRateCalculations.py`.
- Train a Gradient Boosting model using the `Model.py` script.
- Generate final predictions and save the output to the path specified in `config.py`.

---

### 3. Alternative Approach: Using Jupyter Notebooks

You can also run the model in two steps using Jupyter notebooks. This allows you to have more control over each step of the pipeline.

#### **Step 1: Run Feature Engineering Notebook**

1. Open the **Feature Engineering Notebook**.
2. Run all cells to perform data preprocessing and feature engineering. This will generate an intermediate output file containing engineered features.
3. Save the output file.

#### **Step 2: Run the Model Development Notebook**

1. Open the **Model Development Notebook**.
2. Load the output file from the Feature Engineering Notebook into the final model cell.
3. Run the remaining cells in the notebook to train the Gradient Boosting model and generate predictions.

This approach gives you more flexibility if you want to inspect or modify the feature engineering or model training steps separately.

---

### 4. Review the Output

After the pipeline or notebooks have completed, the final output will be saved at the location you specified in the `OUTPUT_DATA_PATH` within the `config.py` file, or directly from the Model Development Notebook.

You can open this file with any spreadsheet application or load it back into Python for further analysis.

---
