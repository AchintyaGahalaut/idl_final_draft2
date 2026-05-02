# Project Implementation Plan: NASA C-MAPSS Turbofan Engine RUL Prediction

This plan outlines the steps to build a deep learning model to predict the **Remaining Useful Life (RUL)** of turbofan engines using the C-MAPSS dataset.

## Phase 1: Data Exploration & Understanding (COMPLETED)
- **Objective**: Get a feel for the data, visualize sensor trends, and understand the "Run-to-Failure" nature of the dataset.
- **Completed Tasks**:
  - Loaded training (`train_FD001.txt`) and testing data (`test_FD001.txt`).
  - Identified the 26-column structure (Unit, Cycle, Settings, Sensors).
  - Calculated raw RUL for both datasets.
  - Visualized sensor trends (e.g., Sensor 2) to confirm degradation signals.

## Phase 2: Data Preprocessing (COMPLETED)
- **Objective**: Prepare the raw data for a deep learning model.
- **Completed Tasks**:
  - **Piecewise Linear RUL (Clipping)**: Capped the target RUL at **125 cycles**. This helps the model focus on the degradation phase rather than the "perfect health" phase.
  - **Normalization**: Used `MinMaxScaler` to scale all sensors, settings, and cycles between **0 and 1**. This ensures no single variable "drowns out" the others.
  - **Feature Selection**: Identified and removed **6 constant sensors** (`s1`, `s5`, `s10`, `s16`, `s18`, `s19`) that provide no learning value.
  - **Data Export**: Generated `train_preprocessed.csv` and `test_preprocessed.csv` as the final clean inputs.

## Phase 3: Model Building (LSTM)
- **Objective**: Build a Long Short-Term Memory (LSTM) network to capture time-series trends.
- **Tasks**:
  - **Data Sequencing**: Transform the flat CSV data into "windows" or "sequences" (e.g., 50 cycles per window).
  - **LSTM Architecture**: Define the LSTM layers, hidden size, and dropout (to prevent overfitting).
  - **Loss & Optimizer**: Use Mean Squared Error (MSE) and the Adam optimizer.

## Phase 4: Training & Evaluation (COMPLETED)
- **Objective**: Train the model on the training set and evaluate it on the test set.
- **Completed Tasks**:
  - Trained the model for 10 epochs as a baseline.
  - Achieved a baseline **RMSE of 40.06**.
  - Verified evaluation logic on the test set.

## Phase 5: Refinement & Reporting (IN PROGRESS)
- **Objective**: Improve the model and summarize results.
- **Tasks**:
  - **Epoch Increase**: Currently training for **100 epochs** to improve accuracy.
  - **Next Up**: Compare the new RMSE with our baseline of 40.06.
  - Visualize predictions vs. actual RUL for the refined model.
