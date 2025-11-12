from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from TDAGENE import GAT
from TDAGENE import compute_tda_feature
from scGNN import preprocess_data, plot_umap_by_time_and_markers
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse
import matplotlib.pyplot as plt
try:
    import scanpy as sc
except Exception:
    sc = None
import networkx as nx

# 可选：用于聚类的 KMeans（当 Leiden/Louvain 不可用时兜底）
try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=7.5e-4,help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=list, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=32, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=128, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='MLP', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--decoder_type', type=str, default='MLP', help='Override decoder: dot/cosine/MLP/bilinear')
parser.add_argument('--patience', type=int, default=10, help='Early stopping patience based on AUC')
parser.add_argument('--fusion_type', type=str, default='gate', help="Fusion type: 'gate' (original) or 'film' (new)")
parser.add_argument('--normalize_features', type=bool, default=True, help='Normalize input features')
parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm (<=0 to disable)')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def embed2file(tf_embed, tg_embed, gene_file, tf_path, target_path):
    tf_embed_np = tf_embed.cpu().detach().numpy()
    tg_embed_np = tg_embed.cpu().detach().numpy()

    gene_set = pd.read_csv(gene_file, index_col=0)

    tf_embed_df = pd.DataFrame(tf_embed_np, index=gene_set['Gene'].values)
    tg_embed_df = pd.DataFrame(tg_embed_np, index=gene_set['Gene'].values)

    tf_dir = os.path.dirname(tf_path)
    tg_dir = os.path.dirname(target_path)
    if tf_dir:
        os.makedirs(tf_dir, exist_ok=True)
    if tg_dir:
        os.makedirs(tg_dir, exist_ok=True)

    tf_embed_df.to_csv(tf_path)
    tg_embed_df.to_csv(target_path)

# 可直接使用预处理的 AnnData 来绘图（与现有训练流程独立）
exp_file = r'D:/test2/test-main/Dataset/Benchmark Dataset/Specific Dataset/hESC/TFs+500/BL--ExpressionData.csv'
tf_file = r'D:/test2/test-main/Dataset/Benchmark Dataset/Specific Dataset/hESC/TFs+500/TF.csv'
target_file = r'D:/test2/test-main/Dataset/Benchmark Dataset/Specific Dataset/hESC/TFs+500/Target.csv'

train_file = os.path.join('Splits2', 'Specific', 'hESC 500', 'Train_set.csv')
val_file = os.path.join('Splits2', 'Specific', 'hESC 500', 'Validation_set.csv')
test_file = os.path.join('Splits2', 'Specific', 'hESC 500', 'Test_set.csv')  # 添加测试集路径

tf_embed_path = r'D:\test2\test-main\result_v1\Specific/hESC 500/result_1/Channel1.csv'
target_embed_path = r'D:\test2\test-main\result_v1\Specific/hESC 500/result_1/Channel2.csv'

data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data().astype(np.float32)
tf_indices = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature).float()
tf_indices = torch.from_numpy(tf_indices).long()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
if args.normalize_features:
    data_feature = F.normalize(data_feature, p=2, dim=1)
tf_indices = tf_indices.to(device)

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values  # 加载测试集数据

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf_indices, loop=args.loop)

adj = adj2saprse_tensor(adj).to(device)

# Precompute TDA feature
tda_feature = compute_tda_feature(data_feature, adj, device)

model = GAT(input_dim=feature.size(1),
                 hidden1_dim=args.hidden_dim[0],
                 hidden2_dim=args.hidden_dim[1],
                 hidden3_dim=args.hidden_dim[2],
                 output_dim=args.output_dim,
                 num_head1=args.num_head[0],
                 num_head2=args.num_head[1],
                 alpha=args.alpha,
                 device=device,
                 type=args.Type,
                 reduction=args.reduction,
                 dropout=args.dropout,
                 decoder_type=args.decoder_type,
                 fusion_type=args.fusion_type,
                 extra_feat_dim=1)

model = model.to(device)

# Binary classification default
if not args.flag:
    train_labels = torch.from_numpy(train_data[:, -1]).float().to(device)
    num_pos = max(train_labels.sum().item(), 1.0)
    num_neg = max(len(train_labels) - train_labels.sum().item(), 1.0)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    # Placeholder for multi-class
    loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = 'model/best'
os.makedirs(model_path, exist_ok=True)
best_ckpt = os.path.join(model_path, 'best_model.pkl')

best_auc = -1.0
no_improve = 0

train_loader = DataLoader(train_load, batch_size=args.batch_size, shuffle=True)
num_batches = len(train_loader)

for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in train_loader:
        model.train()
        optimizer.zero_grad()

        train_x = train_x.long().to(device)
        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.float().to(device).view(-1, 1)

        pred_logit = model(data_feature, adj, train_x, tda_feature)
        loss = loss_fn(pred_logit, train_y)
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / max(1, num_batches)

    # Validation
    model.eval()
    with torch.no_grad():
        val_samples = torch.from_numpy(validation_data[:, :2]).long().to(device)
        val_labels = torch.from_numpy(validation_data[:, -1]).float().to(device) if not args.flag else torch.from_numpy(validation_data[:, -1]).to(device)

        val_logits = model(data_feature, adj, val_samples, tda_feature)
        if not args.flag:
            val_probs = torch.sigmoid(val_logits)
            AUC, AUPR, AUPR_norm = Evaluation(y_pred=val_probs, y_true=val_labels.view(-1, 1), flag=args.flag)
        else:
            val_probs = torch.softmax(val_logits, dim=1)
            AUC, AUPR, AUPR_norm = Evaluation(y_pred=val_probs, y_true=val_labels, flag=args.flag)

    print('Epoch:{}'.format(epoch + 1),
          'train loss:{:.4f}'.format(avg_loss),
          'AUC:{:.3f}'.format(AUC),
          'AUPR:{:.3f}'.format(AUPR))

    # Step scheduler once per epoch
    scheduler.step()

    # Early stopping on AUC
    if AUC > best_auc:
        best_auc = AUC
        no_improve = 0
        torch.save(model.state_dict(), best_ckpt)
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load best model and export embeddings
print("Training completed. Loading best model for testing...")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

# Test the model on test set
print("Evaluating on test set...")
with torch.no_grad():
    test_samples = torch.from_numpy(test_data[:, :2]).long().to(device)
    test_labels = torch.from_numpy(test_data[:, -1]).float().to(device) if not args.flag else torch.from_numpy(test_data[:, -1]).to(device)
    
    test_logits = model(data_feature, adj, test_samples, tda_feature)
    if not args.flag:
        test_probs = torch.sigmoid(test_logits)
        test_AUC, test_AUPR, test_AUPR_norm = Evaluation(y_pred=test_probs, y_true=test_labels.view(-1, 1), flag=args.flag)
    else:
        test_probs = torch.softmax(test_logits, dim=1)
        test_AUC, test_AUPR, test_AUPR_norm = Evaluation(y_pred=test_probs, y_true=test_labels, flag=args.flag)

print('Test Results:',
      'AUC:{:.3f}'.format(test_AUC),
      'AUPR:{:.3f}'.format(test_AUPR),
      'Normalized AUPR:{:.3f}'.format(test_AUPR_norm))

# Export embeddings after testing
with torch.no_grad():
    tf_embed, target_embed = model.compute_embeddings(data_feature, adj, tda_feature)
embed2file(tf_embed, target_embed, target_file, tf_embed_path, target_embed_path)
print("Embeddings saved successfully.")

# ---------- 生成 UMAP 图（按时间与 marker 基因着色）----------
try:
    # 复用 scGNN 的预处理，得到 AnnData
    _, _, adata = preprocess_data(exp_file, hvg_n=2000, log_normalize=True, corr_threshold=0.5, infer_time=True)
    umap_out_dir = os.path.join(os.path.dirname(tf_embed_path), 'umap')
    plot_umap_by_time_and_markers(
        adata,
        output_dir=umap_out_dir,
        marker_genes=['NANOG', 'POU5F1', 'SOX2'],
        n_comps=50,
        n_neighbors=10,
        n_pcs=40
    )
    print(f"UMAP figures saved to: {umap_out_dir}")
except Exception as e:
    print(f"UMAP plotting skipped due to error: {e}")

# ---------- 使用训练好的模型嵌入绘图与导出关系 ----------
try:
    # 读取基因名
    gene_set = pd.read_csv(target_file, index_col=0)
    gene_names = gene_set['Gene'].values if 'Gene' in gene_set.columns else np.array([f"g{i}" for i in range(tf_embed.shape[0])])

    # 使用训练后的嵌入构建基因 AnnData 并 UMAP（若可用）
    if sc is not None:
        # 融合 TF/Target 两路嵌入（简单平均）
        gene_embed = (tf_embed.detach().cpu().numpy() + target_embed.detach().cpu().numpy()) / 2.0
        adata_g = sc.AnnData(gene_embed)
        adata_g.obs_names = gene_names
        # PCA 维度需严格小于 min(n_samples, n_features)
        n_samples, n_features = gene_embed.shape
        # 严格小于 min(n_samples, n_features) 防止 arpack 报错
        n_comps_safe = max(2, min(50, n_features - 1, n_samples - 1))
        n_pcs_safe = max(2, min(30, n_features - 1, n_samples - 1, n_comps_safe))
        sc.pp.pca(adata_g, n_comps=n_comps_safe)
        sc.pp.neighbors(adata_g, n_neighbors=15, n_pcs=n_pcs_safe)
        sc.tl.umap(adata_g)
        # 计算聚类：优先 Leiden，其次 Louvain，最后 KMeans 兜底
        clustered = False
        try:
            sc.tl.leiden(adata_g, key_added='cluster', resolution=1.0)
            clustered = True
        except Exception:
            clustered = False

        # 如果没有成功或只有一个簇，则尝试 Louvain
        if not clustered or ('cluster' not in adata_g.obs.columns or pd.unique(adata_g.obs['cluster']).size < 2):
            try:
                sc.tl.louvain(adata_g, key_added='cluster', resolution=1.0)
                clustered = True
            except Exception:
                clustered = False

        # 仍失败或只有一个簇，则使用 KMeans 兜底
        if (not clustered) or ('cluster' not in adata_g.obs.columns or pd.unique(adata_g.obs['cluster']).size < 2):
            if KMeans is not None:
                k_default = max(3, min(12, int(np.sqrt(n_samples))))
                try:
                    km = KMeans(n_clusters=k_default, n_init=20, random_state=seed)
                    labels = km.fit_predict(gene_embed)
                    adata_g.obs['cluster'] = pd.Categorical(labels.astype(str))
                    clustered = True
                except Exception:
                    adata_g.obs['cluster'] = pd.Categorical(['0'] * n_samples)
            else:
                adata_g.obs['cluster'] = pd.Categorical(['0'] * n_samples)

        # 确保为分类类型，便于着色
        if 'cluster' in adata_g.obs.columns:
            if not pd.api.types.is_categorical_dtype(adata_g.obs['cluster']):
                adata_g.obs['cluster'] = adata_g.obs['cluster'].astype('category')

        emb_out_dir = os.path.join(os.path.dirname(tf_embed_path), 'model_embed')
        os.makedirs(emb_out_dir, exist_ok=True)
        # 颜色调板：若簇数 > 20 让 scanpy 自动分配
        n_clusters = pd.unique(adata_g.obs['cluster']).size if 'cluster' in adata_g.obs.columns else 1
        palette = 'tab20' if n_clusters <= 20 else None
        sc.pl.umap(adata_g, color='cluster', palette=palette, title='UMAP of Gene Embeddings (avg TF/Target)', show=False)
        plt.savefig(os.path.join(emb_out_dir, 'umap_gene_embeddings.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model-embedding UMAP saved to: {emb_out_dir}")

    # 基于嵌入的相似度导出热图和Top边
    tf_np = tf_embed.detach().cpu().numpy()
    tg_np = target_embed.detach().cpu().numpy()
    # 归一化后计算余弦相似度矩阵
    tf_norm = tf_np / (np.linalg.norm(tf_np, axis=1, keepdims=True) + 1e-8)
    tg_norm = tg_np / (np.linalg.norm(tg_np, axis=1, keepdims=True) + 1e-8)
    score_mat = tf_norm @ tg_norm.T

    # 保存热图（若维度大，可能较慢）
    try:
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(score_mat, cmap='viridis', cbar=True)
        plt.title('Gene-Gene Score Matrix (cosine on embeddings)')
        heatmap_path = os.path.join(os.path.dirname(tf_embed_path), 'model_embed', 'scores_heatmap.png')
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Score heatmap saved to: {heatmap_path}")
    except Exception:
        pass

    # 导出Top-K边
    topk = 20000
    flat_idx = np.argpartition(-score_mat.ravel(), topk)[:topk]
    rows, cols = np.unravel_index(flat_idx, score_mat.shape)
    top_edges = pd.DataFrame({
        'TF': gene_names[rows],
        'Target': gene_names[cols],
        'Score': score_mat[rows, cols]
    }).sort_values('Score', ascending=False)
    edges_path = os.path.join(os.path.dirname(tf_embed_path), 'model_embed', f'top_edges_{topk}.csv')
    top_edges.to_csv(edges_path, index=False)
    print(f"Top edges saved to: {edges_path}")
except Exception as e:
    print(f"Model-embedding plotting skipped due to error: {e}")


# ---------- 重建 GRN（基于训练后嵌入），并保存边表与核心可视化 ----------
try:
    emb_out_dir = os.path.join(os.path.dirname(tf_embed_path), 'model_embed')
    os.makedirs(emb_out_dir, exist_ok=True)

    # 使用 score_mat；若未生成则计算
    if 'score_mat' not in globals():
        tf_np = tf_embed.detach().cpu().numpy()
        tg_np = target_embed.detach().cpu().numpy()
        tf_norm = tf_np / (np.linalg.norm(tf_np, axis=1, keepdims=True) + 1e-8)
        tg_norm = tg_np / (np.linalg.norm(tg_np, axis=1, keepdims=True) + 1e-8)
        score_mat = tf_norm @ tg_norm.T

    # 选择全局Top-K边构建GRN
    K = 5000
    flat_idx = np.argpartition(-score_mat.ravel(), K)[:K]
    rows, cols = np.unravel_index(flat_idx, score_mat.shape)

    gene_set = pd.read_csv(target_file, index_col=0)
    gene_names = gene_set['Gene'].values if 'Gene' in gene_set.columns else np.array([f"g{i}" for i in range(score_mat.shape[0])])

    grn_df = pd.DataFrame({'TF': gene_names[rows], 'Target': gene_names[cols], 'Score': score_mat[rows, cols]})
    grn_df = grn_df.sort_values('Score', ascending=False)
    grn_path = os.path.join(emb_out_dir, 'reconstructed_grn_edges.csv')
    grn_df.to_csv(grn_path, index=False)
    print(f"Reconstructed GRN edge list saved to: {grn_path}")

    # 构建 NetworkX 图
    G = nx.DiGraph()
    for name in gene_names:
        G.add_node(name)
    for tf, tg, s in grn_df.itertuples(index=False):
        G.add_edge(tf, tg, weight=float(s))

    # 保存图文件（可供 Gephi/Cytoscape 使用）
    gml_path = os.path.join(emb_out_dir, 'reconstructed_grn.gml')
    try:
        nx.write_gml(G, gml_path)
        print(f"GRN GML saved to: {gml_path}")
    except Exception:
        pass

    # 核心圆形可视化：度数最高前 N 个基因
    N = 30
    degrees = dict(G.degree())
    core_nodes = sorted(degrees.keys(), key=lambda n: degrees[n], reverse=True)[:N]
    H = G.subgraph(core_nodes).copy()
    bc = nx.betweenness_centrality(H, normalized=True)
    pos = nx.circular_layout(H)

    import matplotlib as mpl
    fig, ax = plt.subplots(figsize=(8, 8))
    edge_w = [max(0.5, 3.0 * H[u][v]['weight']) for u, v in H.edges()]
    nx.draw_networkx_edges(H, pos, ax=ax, width=edge_w, edge_color='#e67e22', alpha=0.7, arrows=False)
    node_sizes = [3500 * max(0.05, bc.get(n, 0.0)) + 300 for n in H.nodes()]
    nx.draw_networkx_nodes(H, pos, ax=ax, node_size=node_sizes, node_color='#f5cba7', linewidths=1, edgecolors='#b55d12')
    nx.draw_networkx_labels(H, pos, font_size=9, font_color='black')
    ax.set_axis_off()
    ax.set_title('Circular GRN (core genes)')
    out_path = os.path.join(emb_out_dir, 'grn_circular_core.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Circular core GRN saved to: {out_path}")
except Exception as e:
    print(f"GRN reconstruction skipped due to error: {e}")


# ---------- 追加：marker 小提琴图与分组共表达相关性 ----------
try:
    # 复用已预处理的 adata（再次生成，确保与上一步相同设置）
    _, _, adata_v = preprocess_data(exp_file, hvg_n=2000, log_normalize=True, corr_threshold=0.5, infer_time=True)
    out_dir_base = os.path.join(os.path.dirname(tf_embed_path), 'umap')
    os.makedirs(out_dir_base, exist_ok=True)

    markers = ['NANOG', 'POU5F1', 'SOX2']
    # 仅保留存在于数据中的基因
    markers_present = [g for g in markers if g in list(adata_v.var_names)]
    if len(markers_present) == 0:
        raise ValueError('None of the marker genes are present in adata.var_names')

    # 小提琴图（按 time 分组）
    try:
        ax = sc.pl.violin(adata_v, keys=markers_present, groupby='time', multi_panel=True, stripplot=False, jitter=0, show=False)
        plt.savefig(os.path.join(out_dir_base, 'violin_markers_by_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Violin plot saved to: {os.path.join(out_dir_base, 'violin_markers_by_time.png')}")
    except Exception as e_violin:
        print(f"Violin plotting skipped due to error: {e_violin}")

    # 分时间点的共表达相关（Pearson）
    import itertools
    from scipy.stats import pearsonr
    times = sorted(list(pd.unique(adata_v.obs['time']))) if 'time' in adata_v.obs else ['all']
    corr_rows = []
    for t in times:
        ad = adata_v[adata_v.obs['time'] == t, :] if t != 'all' else adata_v
        for g1, g2 in itertools.combinations(markers_present, 2):
            x1 = ad[:, g1].X
            x2 = ad[:, g2].X
            x1 = x1.A.flatten() if hasattr(x1, 'A') else np.array(x1).flatten()
            x2 = x2.A.flatten() if hasattr(x2, 'A') else np.array(x2).flatten()
            try:
                r, p = pearsonr(x1, x2)
            except Exception:
                r, p = np.nan, np.nan
            corr_rows.append({'time': t, 'gene1': g1, 'gene2': g2, 'pearson_r': r, 'p_value': p})

    corr_df = pd.DataFrame(corr_rows)
    corr_csv_path = os.path.join(out_dir_base, 'marker_pairwise_correlation_by_time.csv')
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Marker correlation CSV saved to: {corr_csv_path}")

    # 也绘制每个时间点的 3x3 相关矩阵热图
    try:
        import seaborn as sns
        for t in times:
            ad = adata_v[adata_v.obs['time'] == t, :] if t != 'all' else adata_v
            mat = []
            for g1 in markers_present:
                row = []
                for g2 in markers_present:
                    x1 = ad[:, g1].X
                    x2 = ad[:, g2].X
                    x1 = x1.A.flatten() if hasattr(x1, 'A') else np.array(x1).flatten()
                    x2 = x2.A.flatten() if hasattr(x2, 'A') else np.array(x2).flatten()
                    try:
                        r, _ = pearsonr(x1, x2)
                    except Exception:
                        r = np.nan
                    row.append(r)
                mat.append(row)
            mat = np.array(mat)
            plt.figure(figsize=(4, 3.2))
            sns.heatmap(mat, vmin=-1, vmax=1, cmap='vlag', annot=True, xticklabels=markers_present, yticklabels=markers_present)
            plt.title(f'Correlation (Pearson) - {t}')
            heat_path = os.path.join(out_dir_base, f'corr_matrix_{t}.png')
            plt.savefig(heat_path, dpi=300, bbox_inches='tight')
            plt.close()
        print(f"Per-time correlation heatmaps saved under: {out_dir_base}")
    except Exception as e_heat:
        print(f"Correlation heatmap skipped due to error: {e_heat}")

    # TDA cycles over time（按时间点计算 TDA 的 num_cycles 指标）
    try:
        if 'time' in adata_v.obs:
            time_points = sorted(list(pd.unique(adata_v.obs['time'])))
            tda_num_cycles = []
            def _build_sub_adj(expr_np: np.ndarray, threshold: float = 0.5):
                # expr_np: genes x cells (dense)
                corr = np.corrcoef(expr_np)
                np.fill_diagonal(corr, 0.0)
                mask = corr > threshold
                i, j = np.where(np.triu(mask, k=1))
                if i.size == 0:
                    indices = torch.zeros((2, 0), dtype=torch.long, device=device)
                    values = torch.zeros((0,), dtype=torch.float, device=device)
                    return torch.sparse_coo_tensor(indices, values, (expr_np.shape[0], expr_np.shape[0]), device=device)
                ii = np.concatenate([i, j])
                jj = np.concatenate([j, i])
                indices = torch.tensor(np.vstack((ii, jj)), dtype=torch.long, device=device)
                values = torch.ones(indices.shape[1], dtype=torch.float, device=device)
                return torch.sparse_coo_tensor(indices, values, (expr_np.shape[0], expr_np.shape[0]), device=device).coalesce()
            for tp in time_points:
                sub_adata = adata_v[adata_v.obs['time'] == tp, :]
                sub_x = sub_adata.X.T
                if not isinstance(sub_x, np.ndarray):
                    sub_x = sub_x.A
                # 重算该时间点的共表达邻接，避免与全局 adj 维度不匹配
                sub_adj = _build_sub_adj(sub_x, threshold=0.5)
                sub_x = torch.tensor(sub_x, dtype=torch.float, device=device)
                tda_feat = compute_tda_feature(sub_x, sub_adj, device)
                # 假定 compute_tda_feature 返回向量的第二个元素是 num_cycles
                tda_num_cycles.append(float(tda_feat[1].item()))

            # 将时间点映射为序号以避免字符串排序绘图问题
            xs = list(range(len(time_points)))
            plt.figure(figsize=(7,4))
            plt.scatter(xs, tda_num_cycles, color='blue')
            plt.plot(xs, tda_num_cycles, linestyle='--', color='gray')
            plt.xticks(xs, time_points, rotation=0)
            plt.xlabel('Time Points')
            plt.ylabel('Number of Cycles (TDA)')
            plt.title('TDA Cycles Over Time')
            tda_plot_path = os.path.join(out_dir_base, 'tda_cycles_scatter.png')
            plt.savefig(tda_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"TDA cycles scatter saved to: {tda_plot_path}")
        else:
            print("Skip TDA cycles over time: 'time' not found in adata.obs")
    except Exception as e_tda:
        print(f"TDA cycles plotting skipped due to error: {e_tda}")
except Exception as e:
    print(f"Additional marker plotting skipped due to error: {e}")