# Project Summary: NASA C-MAPSS Turbofan RUL Prediction

**Last Updated**: 2026-04-26
**Current Phase**: Phase 5 (Evaluation & Optimization)

## 🎯 Project Objective
Predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA C-MAPSS dataset. The goal is to build a Deep Learning model that can look at sensor data and estimate how many cycles are left before failure.

## 🛠️ Environment Setup
- **Conda Environment**: `idl`
- **Python Version**: 3.10+
- **Core Libraries Installed**:
  - `pandas`, `numpy`: Data manipulation
  - `matplotlib`, `seaborn`: Visualization
  - `scikit-learn`: Preprocessing
  - `torch`, `torchvision`: Deep Learning (PyTorch)

## 📊 Datasets
The NASA C-MAPSS data is divided into three key file types for each subset (e.g., FD001):

### 1. `train_FD00x.txt` (Full Life Data)
- **Content**: Complete sensor history for multiple engines from start-of-life until **failure**.
- **RUL Calculation**: Since we know exactly when failure occurs (the last row for each unit), we calculate the label manually: $RUL = (MaxCycle_{Unit}) - (CurrentCycle)$.

### 2. `test_FD00x.txt` (Partial Life Data)
- **Content**: Sensor history for a new set of engines, but the data **stops before failure**. 
- **Goal**: This is the input the model uses to predict how much life is remaining.

### 3. `RUL_FD00x.txt` (Ground Truth Answer Key)
- **Content**: Contains one number for each unit in the test file, representing the **additional cycles** left after the test data ends.
- **RUL Calculation**: To label the test rows, we use: $RUL_{Current} = (ValueInRULFile) + (MaxCycle_{UnitInTestFile}) - (CurrentCycle)$.

## 🧠 Model Checkpoints (.pth)
The `lstm_model_FD00x.pth` files store the "learned brain" of the model after training.

### What's Inside?
- **Weights & Biases**: The mathematical patterns discovered by the model.
- **Metadata**: Our scripts also save the `input_size` (number of sensors) and `hidden_size` to ensure the model can be reloaded without structural errors.

### Why Separate Files?
- **Variable Complexity**: FD001 (1 condition) is significantly different from FD004 (6 conditions). A model must be tailored to the specific environment it was trained on.
- **Structural Differences**: Because different datasets have different constant sensors, the "Input Size" (number of plugs) changes. A model built for 14 sensors won't fit a dataset with 21.

## ✅ Progress Log

### 1. Project Planning
- Created an `implementation_plan.md` outlining the 5 phases of development (Exploration, Preprocessing, Modeling, Training, Evaluation).

### 2. Data Understanding & Exploration (Phase 1)
- Analyzed the dataset structure.
- Developed `eda.py` for initial visualization, capturing sensor trends toward failure, and calculating basic RUL (`processed_train_data.csv`, `processed_test_data.csv`).

### 3. Advanced Preprocessing (Phase 2)
- Created `preprocessing.py` to finalize model inputs.
- **Universal Support**: Expanded to support all datasets (`FD001`, `FD002`, `FD003`, `FD004`) via command-line arguments and modular functions.
- Clipped RUL at 125 (Piecewise Linear RUL) to focus the model on the degradation phase.
- **Dynamic Feature Selection**: Identifies and removes constant sensors per-dataset (crucial for multi-condition datasets like FD002/FD004 where sensors vary based on operating settings).
- Normalized sensor values to a [0, 1] range using `MinMaxScaler`.
- Saved final data to dataset-specific files (e.g., `train_preprocessed_FD002.csv`).

### 4. Deep Learning Architecture (Phase 3)
- Developed a standard **LSTM** architecture in `model.py`.
- **Configurable Baseline**: Uses a unidirectional LSTM with a fully connected output layer.
- **Dynamic Input Sizing**: The model automatically adapts to the number of non-constant sensors provided by the preprocessing layer for each dataset.
- `HIDDEN_SIZE` set to 128, `NUM_LAYERS` set to 2.

### 5. Training, Evaluation, & Notebook Integration (Phase 4 & 5)
- Created `train.py` and `evaluate.py` with full parameterization.
- **New Multi-Dataset Notebook (`lstm_multi.ipynb`)**: Consolidates the entire pipeline for all four datasets into a single notebook using a "one cell per file" style.
- **Loss Function**: Maintained **`MSELoss`** as the core baseline for numerical stability and consistency with the original project setup.
- **Checkpointing**: Updated the training script to save comprehensive checkpoints (`lstm_model_{ID}.pth`) containing weights, input dimensions, and hyperparameters.
- **Robust Evaluation**: Added zero-padding to the evaluation logic to handle engines with fewer cycles than the required window size.

## 📂 Key Files
- `instructions.txt`: Core user instructions and requirements.
- `implementation_plan.md`: The full roadmap.
- `preprocessing.py`: Generates dataset-specific preprocessed inputs (Usage: `python preprocessing.py --dataset FD002`).
- `train.py`: Trains the LSTM on a specific dataset (Usage: `python train.py --dataset FD002`).
- `evaluate.py`: Evaluates a trained model on its corresponding test set (Usage: `python evaluate.py --dataset FD002`).
- `model.py`: Stores the PyTorch LSTM class definition.
- `lstm_multi.ipynb`: The main notebook for running the multi-dataset pipeline (Style: One cell per .py file).
- `lstm_model_{ID}.pth`: Saved model checkpoints for each dataset.
- `*.png` (`evaluation_results_{ID}.png`, `training_loss_{ID}.png`): Dataset-specific performance visualizations.

## 🔜 Next Steps
- **Optimization**: The current Test RMSE sits around ~14.
- **Feature Engineering**: To push RMSE below 10, implement moving averages (smoothing) and rolling statistics (Rolling Mean/Std) in the preprocessing pipeline to provide the Bi-LSTM with explicit, denoised trend indicators.
