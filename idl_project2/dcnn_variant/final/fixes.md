# DCNN Pipeline Refactoring Log

This document tracks all the modifications and fixes applied to the `final_dcnn_variant.ipynb` notebook to support multi-dataset execution (FD001-FD004) while strictly adhering to the architecture and parameters defined in Li et al. (2018).

## 1. Modular "Cell-as-Process" Architecture
* **Change:** Restructured the notebook so that each cell acts as an independent execution stage (Preprocessing -> Model Definition -> Training -> Evaluation).
* **Fix:** Removed hardcoded, single-dataset (FD001) procedural code. Replaced it with execution loops at the bottom of each cell that iterate over all four datasets sequentially. Data is now passed between cells using centralized `processed_data` and `trained_models` dictionaries.

## 2. Dynamic Window Sizing ($N_{tw}$)
* **Change:** Parameterized the `process_cmapss_data` function to accept `sequence_length` dynamically.
* **Fix:** The paper specifies different time window sizes based on the dataset's complexity. We implemented a dictionary mapping to enforce these sizes automatically during preprocessing:
  * FD001: 30
  * FD003: 30
  * FD002: 20
  * FD004: 15

## 3. Padding Fix for Convolutional Layers
* **Change:** Used `padding='same'` in the PyTorch `nn.Conv2d` layers.
* **Fix:** Previously, standard padding shrank the time dimension sequentially. With smaller window sizes (like 15 for FD004), the sequence length would collapse before reaching the fully connected layers. Using `padding='same'` ensures the time dimension is preserved throughout the convolutional feature extractors, exactly matching the paper's zero-padding design.

## 4. Helper Function Statelessness
* **Change:** Refactored data generation helpers (`generate_sequences`, `generate_labels`).
* **Fix:** Ensured these functions are completely stateless so they can be reused iteratively inside the dataset loop without causing memory leaks or cross-contamination between FD001-FD004 dataframes.

## 5. Evaluation Workflow and Model Return
* **Change:** Fixed the `train_model` function to return the trained model object.
* **Fix:** Resolved a critical `AttributeError: 'NoneType' object has no attribute 'eval'` in the evaluation cell. The training loop now correctly passes the trained model back to the global `trained_models` dictionary, allowing the test evaluation and plotting functions to run smoothly.

## 6. Persistent Model Saving
* **Change:** Added `torch.save(trained_model.state_dict(), ...)` at the end of the `train.py` execution block.
* **Fix:** Previously, the DCNN notebook only trained models in-memory, requiring a full 250-epoch retraining if the kernel restarted. It now automatically writes `dcnn_model_FD00X.pth` checkpoints directly to the `gem` folder for immediate inference.

## 7. Cleaned Redundant Code
* **Change:** Removed isolated `example usage` code blocks.
* **Fix:** Prevented the preprocessing and training logic from firing prematurely when defining functions. Ensures the notebook reads cleanly top-to-bottom.
