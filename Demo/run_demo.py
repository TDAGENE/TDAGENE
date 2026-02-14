"""
TDAGENE minimal run example (no real dataset required).
Uses small synthetic data to run training and evaluation.
Run from project root: python Demo/run_demo.py
Or from Demo/: python run_demo.py
"""
from __future__ import print_function
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Ensure project modules are importable (run from root or from Demo/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from TDAGENE import GENELink, compute_tda_feature
from utils import scRNADataset, adj2saprse_tensor, Evaluation


def make_synthetic_data(num_genes=60, num_cells=40, num_tfs=15, num_train=300, num_val=80, num_test=80, seed=42):
    """Generate small synthetic data for a quick sanity check."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Expression matrix (genes x cells)
    expr = np.random.randn(num_genes, num_cells).astype(np.float32) * 0.5 + 1.0
    expr = np.clip(expr, 0.1, None)
    df_expr = pd.DataFrame(expr)

    # Normalize (same behavior as load_data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature = scaler.fit_transform(df_expr.T).T.astype(np.float32)
    feature = torch.from_numpy(feature).float()

    # TF indices: first num_tfs genes
    tf_indices = torch.from_numpy(np.arange(num_tfs, dtype=np.int64)).long()

    # Random edges and labels (positive/negative samples)
    def sample_edges(n_edges, pos_ratio=0.5):
        pos = int(n_edges * pos_ratio)
        neg = n_edges - pos
        rows = []
        used = set()
        for _ in range(pos):
            tf, tg = np.random.randint(0, num_tfs), np.random.randint(0, num_genes)
            if tf == tg:
                tg = (tg + 1) % num_genes
            key = (tf, tg)
            if key not in used:
                used.add(key)
                rows.append([tf, tg, 1.0])
        for _ in range(neg):
            tf, tg = np.random.randint(0, num_tfs), np.random.randint(0, num_genes)
            if tf == tg:
                tg = (tg + 1) % num_genes
            key = (tf, tg)
            if key not in used:
                used.add(key)
                rows.append([tf, tg, 0.0])
        return np.array(rows, dtype=np.float32)

    train_set = sample_edges(num_train)
    val_set = sample_edges(num_val)
    test_set = sample_edges(num_test)
    return feature, tf_indices, train_set, val_set, test_set, num_genes


def main():
    print("TDAGENE Demo: quick test with synthetic data")
    print("=" * 50)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    feature, tf_indices, train_set, val_set, test_set, num_genes = make_synthetic_data(
        num_genes=60, num_cells=40, num_tfs=15,
        num_train=300, num_val=80, num_test=80,
        seed=42
    )
    data_feature = feature.to(device)
    data_feature = F.normalize(data_feature, p=2, dim=1)
    tf_indices = tf_indices.to(device)

    # Dataset and adjacency (training edges only)
    train_load = scRNADataset(train_set, num_genes, flag=False)
    adj = train_load.Adj_Generate(tf_indices.cpu().numpy(), loop=False)
    adj = adj2saprse_tensor(adj).to(device)

    tda_feature = compute_tda_feature(data_feature, adj, device)
    print("TDA feature (avg_pers0, num_cycles, ...):", tda_feature.cpu().numpy())

    # Small model for fast demo
    hidden_dim = [64, 32, 16]
    output_dim = 16
    model = GENELink(
        input_dim=feature.size(1),
        hidden1_dim=hidden_dim[0],
        hidden2_dim=hidden_dim[1],
        hidden3_dim=hidden_dim[2],
        output_dim=output_dim,
        num_head1=2,
        num_head2=2,
        alpha=0.2,
        device=device,
        type="MLP",
        reduction="concate",
        dropout=0.1,
        decoder_type="MLP",
        fusion_type="gate",
        extra_feat_dim=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_load, batch_size=64, shuffle=True)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.long().to(device)
            batch_y = batch_y.float().to(device).view(-1, 1)
            optimizer.zero_grad()
            logits = model(data_feature, adj, batch_x, tda_feature)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print("Epoch {}  train_loss: {:.4f}".format(epoch + 1, avg_loss))

    model.eval()
    with torch.no_grad():
        val_x = torch.from_numpy(val_set[:, :2]).long().to(device)
        val_y = torch.from_numpy(val_set[:, -1]).float().to(device)
        val_logits = model(data_feature, adj, val_x, tda_feature)
        val_probs = torch.sigmoid(val_logits)
        AUC, AUPR, AUPR_norm = Evaluation(y_pred=val_probs, y_true=val_y.view(-1, 1), flag=False)
    print("Validation  AUC: {:.4f}  AUPR: {:.4f}  AUPR_norm: {:.4f}".format(AUC, AUPR, AUPR_norm))

    with torch.no_grad():
        test_x = torch.from_numpy(test_set[:, :2]).long().to(device)
        test_y = torch.from_numpy(test_set[:, -1]).float().to(device)
        test_logits = model(data_feature, adj, test_x, tda_feature)
        test_probs = torch.sigmoid(test_logits)
        test_AUC, test_AUPR, test_AUPR_norm = Evaluation(y_pred=test_probs, y_true=test_y.view(-1, 1), flag=False)
    print("Test        AUC: {:.4f}  AUPR: {:.4f}  AUPR_norm: {:.4f}".format(test_AUC, test_AUPR, test_AUPR_norm))

    out_dir = os.path.join(ROOT, "Demo", "output")
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        tf_embed, target_embed = model.compute_embeddings(data_feature, adj, tda_feature)
    tf_np = tf_embed.cpu().numpy()
    tg_np = target_embed.cpu().numpy()
    pd.DataFrame(tf_np).to_csv(os.path.join(out_dir, "Channel1_demo.csv"), index=False)
    pd.DataFrame(tg_np).to_csv(os.path.join(out_dir, "Channel2_demo.csv"), index=False)
    print("Embeddings saved to Demo/output/Channel1_demo.csv, Channel2_demo.csv")
    print("=" * 50)
    print("Demo finished.")


if __name__ == "__main__":
    main()
