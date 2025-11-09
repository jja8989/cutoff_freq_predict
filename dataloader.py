
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SignalDataset(Dataset):
    def __init__(self, npz_path, normalize = True, y_mean = None, y_std = None):
        data = np.load(npz_path, allow_pickle = True)
        self.signals = data["signals"]
        log_cutoffs = np.log10(data['cutoff_freqs'])
        
        self.times = data['times']
        self.y_mean = np.mean(log_cutoffs) if y_mean is None else y_mean
        self.y_std = np.std(log_cutoffs) if y_std is None else y_std
        

        self.targets = (log_cutoffs - self.y_mean) / y_std
        self.normalize = normalize
        
        if "filenames" in data:
            self.filenames = data["filenames"]
        else:
            self.filenames = np.array([f"sample_{i:05d}" for i in range(len(self.signals))])
            
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        sig = np.asarray(self.signals[idx], dtype = np.float32)
        time = np.asarray(self.times[idx], dtype=np.float32)
        
        dt_mean = np.mean(np.gradient(time))
        Fs = 1.0 / dt_mean

        
        if self.normalize:
            mean, std = np.mean(sig), np.std(sig)
            if std > 1e-8:
                sig = (sig-mean) / std
            
        x = torch.tensor(sig, dtype = torch.float32).unsqueeze(0)
        y = torch.tensor(self.targets[idx], dtype = torch.float32)
        fname = self.filenames[idx]
        Fs_feature = torch.tensor(np.log10(Fs), dtype=torch.float32)
        
        return x, Fs_feature, y, fname
            

def collate_variable(batch):
    xs, Fs, ys, fnames = zip(*batch)
    lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
    max_len = max(lengths)
    
    padded = torch.zeros(len(xs), 1, max_len)
    
    for i, x in enumerate(xs):
        padded[i, 0, :x.shape[-1]] = x
        
    ys = torch.stack(ys)
    Fs = torch.stack(Fs)
    
    return padded, Fs, lengths, ys, fnames


def create_dataloaders(train_path, val_path, batch_size=16):
    train_data = np.load(train_path, allow_pickle=True)
    log_cutoffs = np.log10(train_data['cutoff_freqs'])
    y_mean, y_std = np.mean(log_cutoffs), np.std(log_cutoffs)
    print(f"[INFO] Global target mean/std: {y_mean:.4f}, {y_std:.4f}")
    
    train_ds = SignalDataset(train_path, y_mean=y_mean, y_std=y_std)
    val_ds = SignalDataset(val_path, y_mean=y_mean, y_std = y_std)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_variable)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn = collate_variable)
    
    return train_dl, val_dl, y_mean, y_std