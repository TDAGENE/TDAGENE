import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import Network_Statistic
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', type=float, default=0.8, help='the ratio of the training set')
parser.add_argument('--val_ratio', type=float, default=0.1, help='the ratio of the validation set (of total)')
parser.add_argument('--num', type=int, default= 500, help='network scale')
parser.add_argument('--p_val', type=float, default=0.5, help='the position of the target with degree equaling to one')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--net', type=str, default='Specific', help='network type')
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--neg_pos_ratio_train', type=float, default=1.0, help='negative:positive ratio in train')
parser.add_argument('--neg_pos_ratio_val', type=float, default=1.0, help='negative:positive ratio in val')
parser.add_argument('--neg_pos_ratio_test', type=float, default=1.0, help='negative:positive ratio in test (ignored for Specific hard-negative mode)')
args = parser.parse_args()



def _print_split_stats(name, df):
    pos = int((df['Label'] == 1).sum())
    neg = int((df['Label'] == 0).sum())
    total = len(df)
    print(f"{name}: total={total}, pos={pos}, neg={neg}, pos_ratio={pos/total if total>0 else 0:.3f}")


def train_val_test_set(label_file,Gene_file,TF_file,train_set_file,val_set_file,test_set_file,density,p_val=args.p_val):

    rs = np.random.RandomState(args.seed)

    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    label = pd.read_csv(label_file, index_col=0)
    tf = label['TF'].values

    tf_list = np.unique(tf)
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    train_pos = {}
    val_pos = {}
    test_pos = {}

    for k in pos_dict.keys():
        cur_pos = list(pos_dict[k])
        rs.shuffle(cur_pos)
        if len(cur_pos) <= 1:
            # place into train or test based on p_val
            if rs.uniform(0,1) <= p_val:
                train_pos[k] = cur_pos
            else:
                test_pos[k] = cur_pos
        else:
            n_total = len(cur_pos)
            n_train = int(round(n_total * args.ratio))
            n_val = int(round(n_total * args.val_ratio))
            n_train = max(0, min(n_total, n_train))
            n_val = max(0, min(n_total - n_train, n_val))
            n_test = n_total - n_train - n_val
            train_pos[k] = cur_pos[:n_train]
            val_pos[k] = cur_pos[n_train:n_train + n_val]
            test_pos[k] = cur_pos[n_train + n_val:]

    # negatives per split respecting ratios and no leakage across splits
    def sample_negatives(pos_map, neg_pos_ratio):
        neg_map = {}
        for k in pos_map.keys():
            neg_map[k] = []
            needed = int(np.ceil(len(pos_map[k]) * neg_pos_ratio))
            used = set(pos_dict.get(k, [])) | {k}
            # sample without replacement when possible
            candidates = np.setdiff1d(gene_set, list(used))
            if len(candidates) == 0 or needed == 0:
                continue
            if needed <= len(candidates):
                sel = rs.choice(candidates, size=needed, replace=False)
            else:
                sel = rs.choice(candidates, size=needed, replace=True)
            neg_map[k].extend(sel.tolist())
        return neg_map

    train_neg = sample_negatives(train_pos, args.neg_pos_ratio_train)
    val_neg = sample_negatives(val_pos, args.neg_pos_ratio_val)

    # Build sets
    train_pos_set, train_neg_set = [], []
    for k in train_pos.keys():
        for j in train_pos[k]:
            train_pos_set.append([k, j])
    for k in train_neg.keys():
        for j in train_neg[k]:
            train_neg_set.append([k, j])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]
    train_sample = train_set.copy()
    for i, val in enumerate(train_sample):
        val.append(train_label[i])
    train = pd.DataFrame(train_sample, columns=['TF', 'Target', 'Label'])
    train.to_csv(train_set_file)

    val_pos_set, val_neg_set_list = [], []
    for k in val_pos.keys():
        for j in val_pos[k]:
            val_pos_set.append([k, j])
    for k in val_neg.keys():
        for j in val_neg[k]:
            val_neg_set_list.append([k, j])

    val_set = val_pos_set + val_neg_set_list
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set_list))]
    val_set_a = np.array(val_set)
    val_sample = pd.DataFrame()
    if len(val_set_a) > 0:
        val_sample['TF'] = val_set_a[:,0]
        val_sample['Target'] = val_set_a[:,1]
    else:
        val_sample['TF'] = []
        val_sample['Target'] = []
    val_sample['Label'] = val_label
    val_sample.to_csv(val_set_file)

    # Test set: keep original density-driven negatives count if provided; otherwise use ratio
    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k, j])

    count = 0
    for k in test_pos.keys():
        count += len(test_pos[k])

    if density is not None and density > 0:
        test_neg_num = int(count // density - count)
    else:
        test_neg_num = int(np.ceil(count * args.neg_pos_ratio_test))

    test_neg_set = []
    if test_neg_num > 0:
        used_pairs = set(map(tuple, train_set + val_set + test_pos_set))
        attempts = 0
        while len(test_neg_set) < test_neg_num and attempts < test_neg_num * 10:
            t1 = rs.choice(tf_set)
            t2 = rs.choice(gene_set)
            pair = (t1, t2)
            if t1 != t2 and pair not in used_pairs:
                test_neg_set.append([t1, t2])
                used_pairs.add(pair)
            attempts += 1

    test_pos_label = [1 for _ in range(len(test_pos_set))]
    test_neg_label = [0 for _ in range(len(test_neg_set))]

    test_set = test_pos_set + test_neg_set
    test_label = test_pos_label + test_neg_label
    for i, val in enumerate(test_set):
        val.append(test_label[i])

    test_sample = pd.DataFrame(test_set, columns=['TF', 'Target', 'Label'])
    test_sample.to_csv(test_set_file)

    # Print stats
    _print_split_stats('Train', train)
    _print_split_stats('Val', val_sample)
    _print_split_stats('Test', test_sample)


def Hard_Negative_Specific_train_test_val(label_file, Gene_file, TF_file, train_set_file,val_set_file,test_set_file,
                                          ratio=args.ratio, p_val=args.p_val):
    rs = np.random.RandomState(args.seed)

    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(Gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(TF_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []

    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)

        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        cur_pos = list(pos_dict[k])
        rs.shuffle(cur_pos)
        if len(cur_pos) ==1:
            if rs.uniform(0,1) <= p_val:
                train_pos[k] = cur_pos
            else:
                test_pos[k] = cur_pos

        elif len(cur_pos) ==2:
            rs.shuffle(cur_pos)
            train_pos[k] = [cur_pos[0]]
            test_pos[k] = [cur_pos[1]]
        else:
            n_total = len(cur_pos)
            n_train = int(round(n_total * args.ratio))
            n_val = int(round(n_total * args.val_ratio))
            n_train = max(0, min(n_total, n_train))
            n_val = max(0, min(n_total - n_train, n_val))
            train_pos[k] = cur_pos[:n_train]
            val_pos[k] = cur_pos[n_train:n_train + n_val]
            test_pos[k] = cur_pos[n_train + n_val:]

    train_neg = {}
    val_neg = {}
    test_neg = {}
    for k in pos_dict.keys():
        neg_candidates = list(neg_dict[k])
        rs.shuffle(neg_candidates)
        neg_num = len(neg_candidates)
        # proportional split of negatives using same ratios as positives
        n_train = int(round(neg_num * args.ratio))
        n_val = int(round(neg_num * args.val_ratio))
        n_train = max(0, min(neg_num, n_train))
        n_val = max(0, min(neg_num - n_train, n_val))
        train_neg[k] = neg_candidates[:n_train]
        val_neg[k] = neg_candidates[n_train:n_train + n_val]
        test_neg[k] = neg_candidates[n_train + n_val:]

    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k,val])

    train_neg_set = []
    for k in train_neg.keys():
        # sample negatives to match desired neg:pos (cap by available)
        need = int(np.ceil(len(train_pos.get(k, [])) * args.neg_pos_ratio_train))
        cand = train_neg[k]
        if need > 0 and len(cand) > 0:
            sel = cand if need >= len(cand) else cand[:need]
            for val in sel:
                train_neg_set.append([k,val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]

    train_sample = np.array(train_set)
    train = pd.DataFrame()
    if len(train_sample) > 0:
        train['TF'] = train_sample[:, 0]
        train['Target'] = train_sample[:, 1]
    else:
        train['TF'] = []
        train['Target'] = []
    train['Label'] = train_label
    train.to_csv(train_set_file)

    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k,val])

    val_neg_set = []
    for k in val_neg.keys():
        need = int(np.ceil(len(val_pos.get(k, [])) * args.neg_pos_ratio_val))
        cand = val_neg[k]
        if need > 0 and len(cand) > 0:
            sel = cand if need >= len(cand) else cand[:need]
            for val in sel:
                val_neg_set.append([k,val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    if len(val_sample) > 0:
        val['TF'] = val_sample[:, 0]
        val['Target'] = val_sample[:, 1]
    else:
        val['TF'] = []
        val['Target'] = []
    val['Label'] = val_label
    val.to_csv(val_set_file)

    test_pos_set = []
    for k in test_pos.keys():
        for j in test_pos[k]:
            test_pos_set.append([k,j])

    test_neg_set = []
    for k in test_neg.keys():
        # keep more negatives for test; allow all remaining
        for j in test_neg[k]:
            test_neg_set.append([k,j])

    test_set = test_pos_set +test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    if len(test_sample) > 0:
        test['TF'] = test_sample[:,0]
        test['Target'] = test_sample[:,1]
    else:
        test['TF'] = []
        test['Target'] = []
    test['Label'] = test_label
    test.to_csv(test_set_file)

    # stats
    _print_split_stats('Train', train)
    _print_split_stats('Val', val)
    _print_split_stats('Test', test)


if __name__ == '__main__':
    data_type = args.data
    net_type = args.net

    density = Network_Statistic(data_type=data_type, net_scale=args.num, net_type=net_type)

    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'Dataset', 'Benchmark Dataset', f'{net_type} Dataset', data_type, f'TFs+{args.num}')

    TF2file = os.path.join(data_dir, 'TF.csv')
    Gene2file = os.path.join(data_dir, 'Target.csv')
    label_file = os.path.join(data_dir, 'Label.csv')

    split_dir = os.path.join(base_dir, 'Splits', net_type, f'{data_type} {args.num}')
    os.makedirs(split_dir, exist_ok=True)

    train_set_file = os.path.join(split_dir, 'Train_set.csv')
    test_set_file = os.path.join(split_dir, 'Test_set.csv')
    val_set_file = os.path.join(split_dir, 'Validation_set.csv')

    if net_type == 'Specific':
        Hard_Negative_Specific_train_test_val(label_file, Gene2file, TF2file, train_set_file, val_set_file,
                                              test_set_file)
    else:
        train_val_test_set(label_file, Gene2file, TF2file, train_set_file, val_set_file, test_set_file, density)

