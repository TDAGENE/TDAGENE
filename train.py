from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from TDAGENE import GENELink
from TDAGENE import compute_tda_feature
from torch.optim.lr_scheduler import StepLR
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
import pandas as pd
import numpy as np
import random
import os
import argparse
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=7.5e-4, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3, 3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=list, default=[128, 64, 32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=32, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=128, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type', type=str, default='MLP', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction', type=str, default='concate', help='how to integrate multihead attention')
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


exp_file = os.path.join('Dataset', 'Benchmark Dataset', '...', 'hESC', 'TFs+500', 'BL--ExpressionData.csv')
tf_file = os.path.join('Dataset', 'Benchmark Dataset', '...', 'hESC', 'TFs+500', 'TF.csv')
target_file = os.path.join('Dataset', 'Benchmark Dataset', '...', 'hESC', 'TFs+500', 'Target.csv')
train_file = os.path.join('...', '...', 'hESC 500', 'Train_set.csv')
val_file = os.path.join('...', '...', 'hESC 500', 'Validation_set.csv')
test_file = os.path.join('...', '...', 'hESC 500', 'Test_set.csv')
tf_embed_path = os.path.join('...', '...', 'hESC 500', '...', 'Channel1.csv')
target_embed_path = os.path.join('...', '...', 'hESC 500', '...', 'Channel2.csv')

data_input = pd.read_csv(exp_file, index_col=0)
loader = load_data(data_input)
feature = loader.exp_data().astype(np.float32)
tf_indices = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
feature = torch.from_numpy(feature).float()
tf_indices = torch.from_numpy(tf_indices).long()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
if args.normalize_features:
    data_feature = F.normalize(data_feature, p=2, dim=1)
tf_indices = tf_indices.to(device)

train_data = pd.read_csv(train_file, index_col=0).values
validation_data = pd.read_csv(val_file, index_col=0).values
test_data = pd.read_csv(test_file, index_col=0).values

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf_indices, loop=args.loop)
adj = adj2saprse_tensor(adj).to(device)

tda_feature = compute_tda_feature(data_feature, adj, device)

model = GENELink(
    input_dim=feature.size(1),
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
    extra_feat_dim=1,
)
model = model.to(device)

if not args.flag:
    train_labels = torch.from_numpy(train_data[:, -1]).float().to(device)
    num_pos = max(train_labels.sum().item(), 1.0)
    num_neg = max(len(train_labels) - train_labels.sum().item(), 1.0)
    pos_weight = torch.tensor([num_neg / num_pos], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
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

    print('Epoch:{}'.format(epoch + 1), 'train loss:{:.4f}'.format(avg_loss), 'AUC:{:.3f}'.format(AUC), 'AUPR:{:.3f}'.format(AUPR))
    scheduler.step()
    if AUC > best_auc:
        best_auc = AUC
        no_improve = 0
        torch.save(model.state_dict(), best_ckpt)
    else:
        no_improve += 1
        if no_improve >= args.patience:
            print("Early stopping triggered after {} epochs".format(epoch + 1))
            break

print("Training completed. Loading best model for testing...")
model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

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

print('Test Results:', 'AUC:{:.3f}'.format(test_AUC), 'AUPR:{:.3f}'.format(test_AUPR), 'Normalized AUPR:{:.3f}'.format(test_AUPR_norm))

with torch.no_grad():
    tf_embed, target_embed = model.compute_embeddings(data_feature, adj, tda_feature)
embed2file(tf_embed, target_embed, target_file, tf_embed_path, target_embed_path)
print("Embeddings saved successfully.")

emb_out_dir = os.path.join(os.path.dirname(tf_embed_path), 'model_embed')
os.makedirs(emb_out_dir, exist_ok=True)
gene_set = pd.read_csv(target_file, index_col=0)
gene_names = gene_set['Gene'].values if 'Gene' in gene_set.columns else np.array([f"g{i}" for i in range(tf_embed.shape[0])])
tf_np = tf_embed.detach().cpu().numpy()
tg_np = target_embed.detach().cpu().numpy()
tf_norm = tf_np / (np.linalg.norm(tf_np, axis=1, keepdims=True) + 1e-8)
tg_norm = tg_np / (np.linalg.norm(tg_np, axis=1, keepdims=True) + 1e-8)
score_mat = tf_norm @ tg_norm.T
topk = 20000
flat_idx = np.argpartition(-score_mat.ravel(), topk)[:topk]
rows, cols = np.unravel_index(flat_idx, score_mat.shape)
top_edges = pd.DataFrame({'TF': gene_names[rows], 'Target': gene_names[cols], 'Score': score_mat[rows, cols]}).sort_values('Score', ascending=False)
top_edges.to_csv(os.path.join(emb_out_dir, 'top_edges_{}.csv'.format(topk)), index=False)
print("Top edges saved.")

K = 5000
flat_idx = np.argpartition(-score_mat.ravel(), K)[:K]
rows, cols = np.unravel_index(flat_idx, score_mat.shape)
grn_df = pd.DataFrame({'TF': gene_names[rows], 'Target': gene_names[cols], 'Score': score_mat[rows, cols]}).sort_values('Score', ascending=False)
grn_path = os.path.join(emb_out_dir, 'reconstructed_grn_edges.csv')
grn_df.to_csv(grn_path, index=False)
print("Reconstructed GRN edge list saved to:", grn_path)
G = nx.DiGraph()
for name in gene_names:
    G.add_node(name)
for tf, tg, s in grn_df.itertuples(index=False):
    G.add_edge(tf, tg, weight=float(s))
gml_path = os.path.join(emb_out_dir, 'reconstructed_grn.gml')
try:
    nx.write_gml(G, gml_path)
    print("GRN GML saved to:", gml_path)
except Exception:
    pass
