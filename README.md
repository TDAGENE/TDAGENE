# TDAGENE: Topology-aware Deep Graph Network for Gene Regulatory Network Inference

TDAGENE is a **topology-aware graph neural network model** designed for single-cell/transcriptomic data, used to reconstruct gene regulatory networks (GRNs) from gene expression profiles. This repository uses **`train.py`** as the main entry point and only includes training, evaluation, and result export, without any plotting, making it convenient for reproduction and further development.

---

## 1. Model Overview

TDAGENE treats genes as nodes and regulatory relationships as edges, integrating:

- **Expression Features**: Derived from the expression matrix (genes × cells/samples);
- **Graph Structural Features**: Propagate information over the TF–Target graph via GAT;
- **TDA Features**: Extract topological features (such as connectivity and number of loops) from graphs built **only from the training set**, enhancing high-order structure modeling.

The model outputs scores for TF–Target edges, used for GRN reconstruction and edge ranking.

---

## 2. Environment and Dependencies

- Python 3.7+
- Install dependencies: `pip install -r requirements.txt`

Main dependencies: `torch`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `networkx`. 
Run `train.py`

---

## 3. Data Preparation

Place the data according to the following directory structure (all paths are relative for easy cross-environment operation):

- **Expression and Gene List** 
  - `Dataset/Benchmark Dataset/Specific Dataset/hESC/TFs+500/BL--ExpressionData.csv`  
- In the same directory: `TF.csv` and `Target.csv` (must include columns `Gene` and `index`).
- **Split File**
  - `.../.../hESC 500/Train_set.csv`  
  - `.../.../hESC 500/Validation_set.csv`  
  - `.../.../hESC 500/Test_set.csv`  

Each line format: TF index, Target index, label (0/1). If using another dataset, you need to modify the corresponding path variables in `train.py`.

---

## 4. Operating Modes

Execute in the project root directory:

```bash
python train.py
```

Common parameter examples:

```bash
python train.py \
  --lr 7.5e-4 \
  --epochs 100 \
  --batch_size 128 \
  --decoder_type MLP \
  --fusion_type gate \
  --patience 10
```

Main parameter description:

| Parameter | Default | Description |
|------|--------|------|
| `--lr` | 7.5e-4 | Learning Rate |
| `--epochs` | 100 | Maximum number of training epochs |
| `--batch_size` | 128 | Batch size |
| `--decoder_type` | MLP | Decoder: dot / cosine / MLP / bilinear |
| `--fusion_type` | gate | TDA and GAT fusion method: gate / film |
| `--patience` | 10 | Validation set AUC early stopping patience value |
| `--hidden_dim` | [128,64,32] | Hidden layer dimensions |
| `--output_dim` | 32 | Embedding dimension |
| `--normalize_features` | True | Whether to L2 normalize the input features |
| `--grad_clip` | 1.0 | Gradient clipping (≤0 means no clipping) |

---

## 5. Output Results

After the run is complete, the following files will be generated:

- **Embedding**
  - `.../Specific/hESC 500/.../Channel1.csv`（TF Embedding）  
  - `.../Specific/hESC 500/.../Channel2.csv`（Target Embedding）
- **model**  
  - `model/best/best_model.pkl`（Weights corresponding to the best validation AUC）
- **Edges and Graphs** (directory `.../.../.../model_embed/`)  
  - `top_edges_20000.csv`：Top 20,000 edges sorted by embedding cosine similarity (TF, Target, Score)  
  - `reconstructed_grn_edges.csv`：Edge table of GRN composed of Top-5000 edges  
  - `reconstructed_grn.gml`：The same GRN GML file can be imported into tools like Cytoscape/Gephi.

The terminal will print: the train loss for each epoch, validation AUC/AUPR; early stopping information; test set AUC, AUPR, and Normalized AUPR.

---

## 6. Code Structure

| File | Description |
|------|------|
| `train.py` | Main entry: data loading, training, validation, testing, embedding, and GRN export (no plotting) |
| `TDAGENE.py` | Model `GENELink` and `compute_tda_feature` |
| `utils.py` | `scRNADataset`, `load_data`, `adj2saprse_tensor`, `Evaluation`, etc. |

---

## 7. TDA and Information Leakage Explanation

The graphs used for TDA features **are constructed only from training set edges**, without using validation/test edges:

- After splitting the data into Train/Validation/Test, only the **Train** part is used to generate the adjacency matrix `adj`;
- `tda_feature = compute_tda_feature(data_feature, adj, device)` relies only on this training graph and expression features.

Therefore, there is no information leakage from injecting test set structural information into TDA.

---

## 8. References and Connections

If this code is helpful for your research, please cite the corresponding TDAGENE paper in your publication. If you have any questions or requests for extensions, feel free to provide feedback via the repository Issues or contact the paper's authors.
