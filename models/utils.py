import numpy as np
import torch

# Early stops the training if validation score doesn't improve after a given patience. Optionally saves model params if
# 'checkpoint_file' is specified.
class EarlyStopping:
    def __init__(self, patience, verbose=False, higher_better=False, delta=0.0, checkpoint_file=None,
                 print_file=None, float_fmt='.6f'):
        self.patience = patience
        self.verbose = verbose
        self.higher_better = higher_better
        self.checkpoint_file = checkpoint_file
        assert delta >= 0.0, "negative 'delta' not allowed"
        self.delta = delta if higher_better else -delta
        self.print_file = print_file
        self.float_fmt = float_fmt
        self.counter = 0
        self.early_stop = False
        # initial value is worst possible value
        self.best_score = -np.inf if higher_better else np.inf

    def __call__(self, score, model, epoch):
        threshold = self.best_score + self.delta
        message_checkpoint = ''
        if self.higher_better:
            improved = score > threshold
        else:
            improved = score < threshold
        if improved:
            if self.verbose:
                #msg = f'Score improved: {self.best_score:{self.float_fmt}} --> {score:{self.float_fmt}} - saving model'
                msg = f'{">" if self.higher_better else "<"} {self.best_score:{self.float_fmt}} --> checkpoint'
                #print(msg, file=self.print_file)
                message_checkpoint = msg
            if self.checkpoint_file:
                self.save_checkpoint(score, model, epoch)
            self.best_score = score
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True
        self.counter += 1
        return self.early_stop, improved, message_checkpoint

    def save_checkpoint(self, score, model, epoch):
        torch.save(model.state_dict(), self.checkpoint_file)
