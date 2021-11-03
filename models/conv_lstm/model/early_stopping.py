import os
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, epoch, model, cfg, idx, count):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, cfg, idx, count)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, epoch, model, cfg, idx, count)
            self.counter = 0

    def save_checkpoint(self, val_loss, epoch, model, cfg, idx, count):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
            'cfg': cfg,
        }
        torch.save(state, os.path.join(self.path, f"cnn_lstm_time_{idx}_{count}.pth"))
        self.val_loss_min = val_loss