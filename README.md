# Quantum Agriculture Forecasting Project

## Overview

A hybrid quantum-classical machine learning approach for agricultural yield prediction using environmental features. This project leverages quantum computing techniques through PennyLane to enhance traditional forecasting methods.

## Features

- Quantum-enhanced feature processing
- Multiple quantum encoding strategies
- Hybrid classical-quantum model training
- Comprehensive evaluation metrics
- Configurable pipeline parameters

## Requirements

- Python 3.x
- PennyLane
- NumPy
- Scikit-learn
- Pandas

## Project Structure

```
aiquest/
├── data/
│   └── ai_features.csv
├── models/
│   ├── classical/
│   └── quantum/
├── results/
├── plots/
├── main.py
├── step1_load_data.py
├── step2_select_features.py
├── step3_quantum_encoding.py
├── step4_quantum_kernel.py
├── step5_train_models.py
├── step6_evaluate_models.py
└── step7_save_results.py
```

## Installation

```bash
git clone https://github.com/yourusername/quantum-agriculture-forecasting.git
cd quantum-agriculture-forecasting
pip install -r requirements.txt
```

## Usage

Run the main script with optional arguments:

```bash
python main.py --input data/ai_features.csv --features 5 --scaling minmax --kernel zz --reps 2
```

### Arguments

- `--input`: Path to input CSV file (default: 'data/ai_features.csv')
- `--features`: Number of top features to select (default: 5)
- `--scaling`: Scaling method ['minmax', 'standard'] (default: 'minmax')
- `--kernel`: Quantum feature map type ['zz', 'pauli'] (default: 'zz')
- `--reps`: Number of circuit repetitions (default: 2)

## Pipeline Steps

1. Data Loading & Preprocessing
2. Feature Selection & Scaling
3. Quantum Feature Encoding
4. Quantum Kernel Construction
5. Model Training
6. Model Evaluation
7. Results Storage

## Current Performance

Based on initial results:

- MSE: 116,740,298
- RMSE: 10,804
- MAE: 8,571
- R² Score: 0.054

## Future Improvements

- Optimize quantum circuit parameters
- Explore alternative encoding strategies
- Implement feature engineering
- Enhance model architecture
- Increase quantum advantage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Contact

DIB Abdelkrim Y T (dibabdelkrimyt98@gmail.com)
MOSTEFAOUI Mohammed (mohammedmostefaoui2@gmail.com)