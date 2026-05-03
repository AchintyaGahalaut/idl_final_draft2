# NASA Turbofan Engine Degradation — Remaining Useful Life Prediction

> **CMU 24-788 Intro to Deep Learning | Final Project | Option 6**

This project uses **deep learning** to predict when aircraft engines will fail — before they actually do. By analyzing streams of sensor data recorded during engine operation, our models learn to estimate the **Remaining Useful Life (RUL)**: the number of cycles an engine has left before it needs maintenance.

---

## 🔍 The Problem: Why Does This Matter?

Imagine an aircraft engine running for thousands of cycles. Inside it, dozens of sensors are constantly measuring temperature, pressure, rotational speed, fuel flow, and more. Over time, the engine degrades — parts wear out, efficiency drops, and eventually it will fail.

Two strategies exist for maintenance:
- **Reactive**: Fix it after it breaks. ❌ Dangerous and expensive.
- **Preventive**: Perform maintenance on a fixed schedule. ❌ Wasteful — you often replace parts that are still healthy.
- **Predictive (our approach)**: Use sensor data to predict *exactly* when maintenance is needed. ✅ Safe and efficient.

This is called **Predictive Maintenance**, and it's one of the most important applications of deep learning in industry. Our goal is to build and compare two different deep learning models that can do this prediction accurately.

---

## 📡 The Dataset: NASA C-MAPSS

We use the **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset, published by NASA. It simulates real turbofan engine degradation under controlled conditions.

### What's in the dataset?
The dataset contains recordings from many different engines. Each engine starts healthy and runs until it fails. The data is split into four separate subsets, each with different levels of complexity:

| Dataset | # Engines (Train) | # Engines (Test) | Operating Conditions | Fault Modes |
|---------|-------------------|-------------------|----------------------|-------------|
| **FD001** | 100 | 100 | 1 | 1 |
| **FD002** | 260 | 259 | 6 | 1 |
| **FD003** | 100 | 100 | 1 | 2 |
| **FD004** | 248 | 248 | 6 | 2 |

- **FD001 and FD003** are "simpler" — all engines operate under the same conditions.
- **FD002 and FD004** are "harder" — engines switch between 6 different flight conditions, making sensor readings noisier and harder to interpret.

### What does each file contain?
Each subset has three files:

1. **`train_FD00x.txt`**: Full sensor histories for multiple engines, from the very first cycle all the way until the engine fails. Since we know the exact failure cycle, we can calculate the "answer" (RUL) for every row.

2. **`test_FD00x.txt`**: Partial sensor histories for a new set of engines — the data cuts off *before* failure. This is what the model has to work with to make its prediction.

3. **`RUL_FD00x.txt`**: The "answer key" — one number per test engine telling us exactly how many cycles are remaining when the test data ends.

### What are the sensor readings?
Each row in the data represents one engine cycle and contains:
- **3 Operational Settings** (e.g., altitude, throttle)
- **21 Sensor Measurements** (e.g., temperature, pressure, rotational speed, fuel flow)

Not all 21 sensors are useful. Some sensors stay constant throughout an engine's life and provide no degradation signal. Our preprocessing step removes these automatically.

### Key Preprocessing Steps
To prepare this raw data for deep learning, we apply several critical transformations:
1. **Dynamic Sensor Selection**: We automatically drop sensors that remain constant (flatline) over time, as they provide no degradation signal.
2. **The "125-Cycle Cap" (Piecewise Linear RUL)**: A key insight from research is that engines don't start degrading immediately. We cap the maximum target RUL at **125 cycles**. This forces the model to ignore the stable "healthy" phase and focus entirely on learning the specific signatures of wear and tear.
3. **Normalization**: We scale all sensor values to a standard range (e.g., `[0, 1]` or `[-1, 1]`) using a Min-Max Scaler fitted strictly on the training data. This ensures high-magnitude sensors don't drown out low-magnitude sensors.
4. **Sliding Windows**: Sequence models need context. We transform the 2D data into 3D "sequences" by extracting sliding windows (e.g., 30 consecutive cycles) to capture the temporal trend leading up to a prediction.

---

## 🧠 The Two Deep Learning Approaches

This project implements and compares two fundamentally different types of neural networks for this task.

---

### Approach 1: LSTM (Long Short-Term Memory) — The "Memory" Network

**Main File: `baseline_lstm/lstm_multi_v2.ipynb`** ⭐

#### What is an LSTM?
A standard neural network looks at one snapshot of data at a time and has no memory of what came before. An **LSTM (Long Short-Term Memory)** network is a special type of **Recurrent Neural Network (RNN)** designed to process *sequences* of data over time — and, crucially, to *remember* important patterns from the past.

Think of it like a person reading a book. A standard network reads one word at a time with no context. An LSTM reads the same word but also "remembers" everything important it read before, allowing it to understand meaning in context.

#### How does it work?
LSTMs achieve this through three special "gates" inside each neuron:
- **Forget Gate**: Decides what old information to throw away.
- **Input Gate**: Decides what new information is worth remembering.
- **Output Gate**: Decides what to pass forward as the output.

This makes LSTMs exceptionally good at detecting long-range patterns in time-series data — like a gradual rise in temperature over 30 cycles that signals engine wear.

#### Our LSTM Architecture:
- **Input**: A 30-cycle window of sensor data (a "snapshot" of the last 30 engine cycles)
- **2 Stacked LSTM Layers**: 128 hidden units each, giving the model deep memory
- **Dropout (20%)**: Randomly disables neurons during training to prevent memorization
- **Output Layer**: A single number — the predicted RUL

---

### Approach 2: DCNN (Deep Convolutional Neural Network) — The "Pattern Scanner"

**Main File: `dcnn_variant/final/final_dcnn_variant.ipynb`** ⭐

#### What is a DCNN?
Convolutional Neural Networks (CNNs) are most famous for recognizing objects in photos. They work by scanning an image with small "filters" that detect specific visual features (edges, colors, shapes). In our project, we apply this same idea to **time-series sensor data**.

Instead of scanning a 2D image, our DCNN scans a **2D matrix of sensor readings over time** (e.g., a 30-cycle × 14-sensor grid). The filters look for characteristic "shapes" in how sensor readings change — patterns that indicate degradation.

#### Why use a CNN for time-series?
A CNN treats the sensor data like an image. Each column is a sensor, each row is a time step. The CNN's filters slide along the **time axis** to detect patterns like:
- A sudden pressure spike
- A gradual temperature drift
- Correlated changes across multiple sensors

This approach was pioneered by **Li et al. (2018)** in the paper *"Remaining Useful Life Estimation in Prognostics Using Deep Convolutional Neural Networks"*, which this notebook directly implements.

#### Our DCNN Architecture (from the paper):
- **Input**: A window × 14-sensor "image" of data
- **4 Convolutional Layers**: 10 filters each, with 10×1 kernels (scan 10 time-steps per sensor)
- **1 Summarizing Convolutional Layer**: 1 filter with a 3×1 kernel
- **Tanh Activations**: To capture complex non-linear degradation patterns
- **50% Dropout**: Strong regularization to prevent overfitting
- **2 Fully Connected Layers**: 100 hidden neurons → 1 output (predicted RUL)
- **Xavier Normal Initialization**: Smart weight initialization for stable, fast training

---

## 📊 Final Results & Evaluation Protocol

Both models were evaluated using the strict **"Last-Window-Only"** protocol. Instead of evaluating on every possible sliding window in the test set (which inflates scores), each test engine gets exactly **one prediction** based on its final observed window of data. This perfectly simulates the real-world scenario of predicting RUL at the current moment in time.

### The Metrics
1. **Test RMSE (Root Mean Squared Error)**: Tells us the average number of cycles the model was wrong by.
2. **NASA Asymmetric Score**: In aviation, predicting an engine has *more* life than it actually does (late prediction) is incredibly dangerous. Predicting it has *less* life (early prediction) is just wasteful. The official NASA scoring function penalizes late predictions exponentially more than early predictions. 

### Final Performance (RMSE — Lower is Better)

| Dataset | LSTM Baseline | Deep CNN Variant | Winner |
|---------|--------------|-----------------|--------|
| **FD001** | 13.41 | **13.14** | DCNN ✅ |
| **FD002** | **20.81** | 21.04 | LSTM ✅ |
| **FD003** | 12.31 | **11.98** | DCNN ✅ |
| **FD004** | **24.64** | 27.10 | LSTM ✅ |
| **Average** | **17.79** | 18.32 | LSTM ✅ |

### Key Insight
> The DCNN wins on **simple, single-condition** datasets (FD001, FD003), where its convolutional pattern detection excels.
> The LSTM wins on **complex, multi-condition** datasets (FD002, FD004), where its temporal memory helps generalize across different operating environments.
> Despite using **5× more training epochs and a larger batch size**, the DCNN doesn't consistently outperform the much simpler LSTM.

For a full breakdown, see [`model_comparison.md`](model_comparison.md).

---

## 📁 Project Structure

```
idl_project2/
│
├── 📓 baseline_lstm/                    ← LSTM Model (Baseline)
│   ├── lstm_multi_v2.ipynb              ⭐ MAIN NOTEBOOK — Full pipeline for all 4 datasets
│   ├── lstm_model_FD001.pth             ← Saved trained weights for FD001
│   ├── lstm_model_FD002.pth             ← Saved trained weights for FD002
│   ├── lstm_model_FD003.pth             ← Saved trained weights for FD003
│   ├── lstm_model_FD004.pth             ← Saved trained weights for FD004
│   ├── README.md                        ← LSTM-specific technical documentation
│   ├── explanation.md                   ← Plain-English walkthrough of each cell
│   ├── summary.md                       ← High-level project summary and results
│   ├── model.py                         ← Standalone LSTM class definition
│   ├── preprocessing.py                 ← Standalone preprocessing script
│   ├── train.py                         ← Standalone training script
│   └── evaluate.py                      ← Standalone evaluation script
│
├── 📓 dcnn_variant/final/               ← Deep CNN Model (Variant)
│   ├── final_dcnn_variant.ipynb         ⭐ MAIN NOTEBOOK — Full pipeline for all 4 datasets
│   ├── dcnn_model_FD001.pth             ← Saved trained weights for FD001
│   ├── dcnn_model_FD002.pth             ← Saved trained weights for FD002
│   ├── dcnn_model_FD003.pth             ← Saved trained weights for FD003
│   ├── dcnn_model_FD004.pth             ← Saved trained weights for FD004
│   ├── README.md                        ← DCNN-specific technical documentation
│   ├── explanation.md                   ← Plain-English walkthrough of each cell
│   └── fixes.md                         ← Change log of fixes applied
│
├── 📊 model_comparison.md               ← Side-by-side LSTM vs. DCNN results & analysis
├── 📝 project_report.txt                ← Full project write-up
├── 📋 implementation_plan.md            ← Original development roadmap
│
├── 📂 scratch/                          ← Temporary utility scripts (gitignored)
│   ├── update_lstm_plots.py
│   ├── update_lstm_plots_lines.py
│   ├── update_dcnn_plots.py
│   ├── update_dcnn_eval.py
│   ├── scratch_extract_metrics.py
│   └── scratch_extract_text.py
│
├── 📂 6.+Turbofan+Engine+.../           ← Raw NASA C-MAPSS dataset files
│
├── .gitignore                           ← Git ignore rules (excludes scratch/ and large files)
│
└── 📄 Reference Papers (PDF)
    ├── Remaining useful life estimation... (Li et al., 2018) ← DCNN paper
    ├── Predicting Remaining Useful Life using Time Series Embeddings.pdf
    └── Deep learning-based tool wear prediction...pdf
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.10+, PyTorch, pandas, numpy, scikit-learn, matplotlib
- A GPU is strongly recommended (both notebooks were trained on Google Colab)

### Running the LSTM Baseline
1. Open `baseline_lstm/lstm_multi_v2.ipynb` in Jupyter or Google Colab.
2. Run cells in order: **Setup → Preprocessing → Model → Training → Evaluation**.
3. To skip retraining and just see results, run only the **Model** cell (to load the class), then the **Evaluation** cell directly. It will load the saved `.pth` files automatically.

### Running the Deep CNN Variant
1. Open `dcnn_variant/final/final_dcnn_variant.ipynb` in Jupyter or Google Colab.
2. Run cells in order: **Setup → Preprocessing → Model → Training → Evaluation**.
3. Same shortcut applies: run the **Model** cell, then skip straight to **Evaluation**.

---

## 📚 References

- **Dataset**: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage propagation modeling for aircraft engine run-to-failure simulation.* NASA Ames Research Center.
- **DCNN Architecture**: Li, X., Ding, Q., & Sun, J. Q. (2018). *Remaining useful life estimation in prognostics using deep convolutional neural networks.* Reliability Engineering & System Safety.
- **Evaluation Protocol**: Babu, G. S., Zhao, P., & Li, X. L. (2016). *Deep convolutional neural network based regression approach for estimation of remaining useful life.*

---

## 💡 Note on Initial Contributions

The preliminary work, experiments, and initial code for contributions 1 and 2 of this project are preserved in the following notebook for review:
- `contribution_one_and_two/dev_contri_1_2_final.ipynb`

