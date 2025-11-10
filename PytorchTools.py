import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if monitored metric doesn't improve after a given patience."""
    def __init__(self,save_dir, patience=7,verbose=False, delta=0, mode='max'):
        """
        Args:
            patience (int): How long to wait after last time monitored metric improved.
                            Default: 7
            verbose (bool): If True, prints a message for each improvement.
                            Default: False
            delta (float): Minimum change to qualify as an improvement.
                            Default: 0
            mode (str): 'max' to maximize the metric (e.g., AUPR), 'min' to minimize (e.g., loss).
                         Default: 'max'
        """
        self.patience = patience
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        if mode not in ['min','max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

    def __call__(self, score, model):
        # score is the monitored metric (e.g., AUPR for mode='max', or loss for mode='min')
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        else:
            improved = (score > self.best_score + self.delta) if self.mode == 'max' else (score < self.best_score - self.delta)
            if improved:
                self.best_score = score
                self.save_checkpoint(score, model)
                self.counter = 0
            else:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, score, model):
        '''Saves model when monitored metric improves.'''
        if self.verbose:
            print(f'Validation metric improved to ({score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_dir+'.pkl')
        self.val_loss_min = score

