import numpy as np
import torch
from torch.utils.data import Dataset

class VirtualAgentDataset(Dataset):
    def __init__(self, npz_path, split='train'):
        """
        Carica il dataset del Virtual Agent e lo prepara per il Transformer.
        """
        # 1. CARICAMENTO DATI
        data = np.load(npz_path)
        X_endo_all = data['X_endo']
        X_exo_all = data['X_exo']
        X_cond_all = data['X_cond']
        Y_target_all = data['Y_target']

        total_samples = len(X_endo_all)
        
        # 2. SPLITTING (70% Train, 15% Val, 15% Test)
        # Usiamo indici sequenziali per rispettare l'ordine temporale
        train_end = int(total_samples * 0.7)
        val_end = train_end + int(total_samples * 0.15)

        if split == 'train':
            idx_start, idx_end = 0, train_end
        elif split == 'val':
            idx_start, idx_end = train_end, val_end
        elif split == 'test':
            idx_start, idx_end = val_end, total_samples
        else:
            raise ValueError("Il parametro 'split' deve essere 'train', 'val', o 'test'")

        self.X_endo = X_endo_all[idx_start:idx_end]
        self.X_exo = X_exo_all[idx_start:idx_end]
        self.X_cond = X_cond_all[idx_start:idx_end]
        self.Y_target = Y_target_all[idx_start:idx_end]

        # 3. NORMALIZZAZIONE (Z-Score Globale)
        # Regola d'oro: Le statistiche (media e dev. std) si calcolano SOLO sul Train set 
        # per non "sbirciare" nel futuro (Data Leakage), poi si applicano a tutti gli split.
        self.mean_endo = np.mean(X_endo_all[:train_end], axis=(0, 1), keepdims=True)
        self.std_endo = np.std(X_endo_all[:train_end], axis=(0, 1), keepdims=True) + 1e-8

        self.mean_exo = np.mean(X_exo_all[:train_end], axis=(0, 1), keepdims=True)
        self.std_exo = np.std(X_exo_all[:train_end], axis=(0, 1), keepdims=True) + 1e-8

        self.mean_cond = np.mean(X_cond_all[:train_end], axis=(0, 1), keepdims=True)
        self.std_cond = np.std(X_cond_all[:train_end], axis=(0, 1), keepdims=True) + 1e-8

        self.mean_y = np.mean(Y_target_all[:train_end], axis=(0, 1), keepdims=True)
        self.std_y = np.std(Y_target_all[:train_end], axis=(0, 1), keepdims=True) + 1e-8

        # Applichiamo la normalizzazione
        self.X_endo = (self.X_endo - self.mean_endo) / self.std_endo
        self.X_exo = (self.X_exo - self.mean_exo) / self.std_exo
        self.X_cond = (self.X_cond - self.mean_cond) / self.std_cond
        self.Y_target = (self.Y_target - self.mean_y) / self.std_y

        # 4. CONVERSIONE IN TENSORI PYTORCH
        self.X_endo = torch.tensor(self.X_endo, dtype=torch.float32)
        self.X_exo = torch.tensor(self.X_exo, dtype=torch.float32)
        self.X_cond = torch.tensor(self.X_cond, dtype=torch.float32)
        self.Y_target = torch.tensor(self.Y_target, dtype=torch.float32)

        print(f"[{split.upper()}] Caricati {len(self.X_endo)} campioni.")

    def __len__(self):
        return len(self.X_endo)

    def __getitem__(self, idx):
        # I nostri dati sono (Tempo, Canali) -> (100, 8)
        # Il modello vuole (Canali, Tempo) -> (8, 100)
        # Usiamo .permute(1, 0) per scambiare la dimensione 0 e 1 per le feature di input
        
        x_en = self.X_endo[idx].permute(1, 0) 
        x_ex = self.X_exo[idx].permute(1, 0)
        # ---> LA MODIFICA È QUI <---
        # I dati condizionali sono un contesto globale. Prendiamo solo l'ULTIMO frame della finestra.
        # Niente permute, restituiamo un vettore 1D di 9 elementi (che il Dataloader farà diventare Batch x 9)
        x_co = self.X_cond[idx][-1]
        
        # Y_target rimane (Horizon, Canali) perché il Transformer in output vuole così
        y_tar = self.Y_target[idx]

        return x_en, x_ex, x_co, y_tar