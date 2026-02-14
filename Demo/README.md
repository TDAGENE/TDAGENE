# TDAGENE Demo: Quick Start Example

This directory provides simple test scripts that **do not rely on real datasets**, used to verify whether the environment and code are running properly.

## How to Run

Execute in the **project root directory**:

```bash
python Demo/run_demo.py
```

Or execute in the `Demo` directory:

```bash
cd Demo
python run_demo.py
```

## Explanation

- **Data**: The script generates small-scale random data internally (60 genes × 40 cells, about 300 training edges, and 80 validation/test edges), with no need to prepare any data files.
- **Process**: Build the graph and TDA features → Train for 3 epochs → Calculate AUC/AUPR on the validation and test sets → Save the TF/Target embeddings to `Demo/output/`.
- **Output**:
- Terminal output: loss for each round, AUC, AUPR, and Normalized AUPR for Validation/Test.
- Files: `Demo/output/Channel1_demo.csv` and `Channel2_demo.csv` (embedding matrices).

## Dependencies

Consistent with the main project, it requires `torch`, `numpy`, `pandas`, `scikit-learn` to be installed, as well as the `TDAGENE` and `utils` modules from the project (they should be correctly importable from the root directory).

## Expected Results

When it runs successfully, you will see output like this at the end:

```
Validation  AUC: 0.xxxx  AUPR: 0.xxxx  AUPR_norm: x.xxxx
Test        AUC: 0.xxxx  AUPR: 0.xxxx  AUPR_norm: x.xxxx
Embeddings saved to Demo/output/Channel1_demo.csv, Channel2_demo.csv
Demo run completed.
```

Since the data is randomly generated, the metric values have no actual significance and are only used to verify that the process can run smoothly.
