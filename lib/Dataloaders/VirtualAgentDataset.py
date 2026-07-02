import numpy as np
import torch
from torch.utils.data import Dataset

class VirtualAgentDataset(Dataset):
    def __init__(self, npz_path, split='train', train_ratio=0.7, val_ratio=0.15, norm='zscore'):
        """
        Carica, formatta e normalizza il dataset del Virtual Agent.
        
        Parametri:
        - npz_path: percorso al file VA_Dataset_Tensor.npz
        - split: 'train', 'val' o 'test'
        - train_ratio: percentuale di dati usati per il training (default 70%)
        - val_ratio: percentuale di dati usati per la validazione (default 15%)
        - norm: tipo di normalizzazione (es. 'zscore' o None)
        """
        self.split = split
        self.norm = norm
        
        # 1. Caricamento del Tensore Grezzo
        data = np.load(npz_path)
        
        # 2. FIX DIMENSIONALE DEFINITIVO (Cruciale per il Transformer!)
        # Il file .npz ha forma (Batch, Tempo, Canali).
        # Il modello PyTorch VUOLE (Batch, Canali, Tempo).
        X_endo_full = np.transpose(data['X_endo'], (0, 2, 1))   # (N, 8, lookback)
        X_exo_full = np.transpose(data['X_exo'], (0, 2, 1))     # (N, 18, lookback)
        Y_target_full = np.transpose(data['Y_target'], (0, 2, 1)) # (N, 1, horizon)
        
        # FIX PER I DATI CONDIZIONALI (cond): Vettore statico, prendiamo l'ultimo frame
        X_cond_full = data['X_cond'][:, -1, :]                  # (N, 9)
        
        # 3. SPLIT DATASET
        total_samples = len(X_endo_full)
        train_end = int(total_samples * train_ratio)
        val_end = int(total_samples * (train_ratio + val_ratio))
        
        if split == 'train':
            idx_start, idx_end = 0, train_end
        elif split == 'val':
            idx_start, idx_end = train_end, val_end
        elif split == 'test':
            idx_start, idx_end = val_end, total_samples
        else:
            raise ValueError("Lo split deve essere 'train', 'val' o 'test'.")
            
        self.X_endo = X_endo_full[idx_start:idx_end]
        self.X_exo = X_exo_full[idx_start:idx_end]
        self.X_cond = X_cond_full[idx_start:idx_end]
        self.Y_target = Y_target_full[idx_start:idx_end]
        
        # 4. NORMALIZZAZIONE (Z-SCORE INTELLIGENTE)
        if self.norm == 'zscore':
            # Statistiche calcolate SOLO sul TRAIN per evitare Data Leakage
            train_X_endo = X_endo_full[:train_end] 
            train_X_exo = X_exo_full[:train_end]   
            train_X_cond = X_cond_full[:train_end]
            
            # --- 4.1 ENDO (Distanze 3D) ---
            # Troviamo i valori validi: escludiamo i giocatori assenti (che ora sono -1.0!)
            valid_train_endo = train_X_endo[train_X_endo != -1.0]
            self.endo_mean = np.mean(valid_train_endo) if len(valid_train_endo) > 0 else 0.0
            self.endo_std = np.std(valid_train_endo) + 1e-8 if len(valid_train_endo) > 0 else 1.0
            
            # Applichiamo la normalizzazione SOLO ai giocatori presenti.
            # Convertiamo i -1.0 in 0.0 per la rete neurale (che in Z-score significa "neutrale")
            endo_mask = self.X_endo != -1.0
            self.X_endo = np.where(endo_mask, (self.X_endo - self.endo_mean) / self.endo_std, 0.0)
            
            # Stessa cosa per il target (per sicurezza estrema, anche se l'estrattore li ha già filtrati)
            target_mask = self.Y_target != -1.0
            self.Y_target = np.where(target_mask, (self.Y_target - self.endo_mean) / self.endo_std, 0.0)
            
            # --- 4.2 EXO (18 Variabili ambientali) ---
            # Mascheriamo i 999.0 impostati da Extract_VA trasformandoli in NaN temporanei
            exo_train_nan = np.where(train_X_exo >= 900.0, np.nan, train_X_exo)
            
            # Calcoliamo medie separate per ogni canale (ignorando i NaN!)
            self.exo_mean = np.nanmean(exo_train_nan, axis=(0, 2), keepdims=True)
            self.exo_std = np.nanstd(exo_train_nan, axis=(0, 2), keepdims=True) + 1e-8
            
            # Rimuoviamo eventuali NaN rimasti se un canale era interamente vuoto
            self.exo_mean = np.nan_to_num(self.exo_mean, nan=0.0)
            self.exo_std = np.nan_to_num(self.exo_std, nan=1.0)
            
            # Normalizziamo solo i valori reali. I 999.0 verranno rimpiazzati da uno 0.0 piatto e inoffensivo!
            valid_exo_mask = self.X_exo < 900.0
            self.X_exo = np.where(valid_exo_mask, (self.X_exo - self.exo_mean) / self.exo_std, 0.0)
            
            # --- 4.3 COND (Variabili Statiche) ---
            # RowSide (indici 0-7) restano raw: -1, 0, 1
            # Normalizziamo solo il TeamScore (indice 8) (i 30.0 vuoti faranno media normalmente)
            train_score = train_X_cond[:, 8]
            self.score_mean = np.mean(train_score)
            self.score_std = np.std(train_score) + 1e-8
            
            self.X_cond[:, 8] = (self.X_cond[:, 8] - self.score_mean) / self.score_std
            
        # 5. TENSORIZZAZIONE
        self.X_endo = torch.tensor(self.X_endo, dtype=torch.float32)
        self.X_exo = torch.tensor(self.X_exo, dtype=torch.float32)
        self.X_cond = torch.tensor(self.X_cond, dtype=torch.float32)
        self.Y_target = torch.tensor(self.Y_target, dtype=torch.float32)

        print(f"[{split.upper()}] Creato con {len(self.X_endo)} campioni. (Norm: {self.norm})")

    def __len__(self):
        return len(self.X_endo)

    def __getitem__(self, idx):
        return self.X_endo[idx], self.X_exo[idx], self.X_cond[idx], self.Y_target[idx]