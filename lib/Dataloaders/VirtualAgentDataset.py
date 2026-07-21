# # import numpy as np
# # import torch
# # from torch.utils.data import Dataset

# # class VirtualAgentDataset(Dataset):
# #     def __init__(self, npz_path, split='train', train_ratio=0.7, val_ratio=0.15, norm=None):
# #         """
# #         Carica e formatta il dataset del Virtual Agent.
        
# #         Parametri:
# #         - npz_path: percorso al file VA_Dataset_Tensor.npz
# #         - split: 'train', 'val' o 'test'
# #         - train_ratio: percentuale di dati usati per il training (default 70%)
# #         - val_ratio: percentuale di dati usati per la validazione (default 15%)
# #         - Nota: La normalizzazione fisica avviene a monte in Extract_VA.py.
# #             Il Dataloader si occupa solo di mappare le assenze (-1.0) a zero (0.0).
# #         """
# #         self.split = split
# #         self.norm = norm
        
# #         print(f"[{split.upper()}] Caricamento array dal disco in corso (Attendi qualche secondo)...")
# #         # 1. Caricamento del Tensore Grezzo
# #         data = np.load(npz_path)

# #         # Troviamo il numero totale di campioni
# #         total_samples = data['X_endo'].shape[0]
        
# #         # 2. FIX DIMENSIONALE DEFINITIVO (Cruciale per il Transformer!)
# #         # Il file .npz ha forma (Batch, Tempo, Canali).
# #         # Il modello PyTorch VUOLE (Batch, Canali, Tempo).
# #         X_endo_full = np.transpose(data['X_endo'], (0, 2, 1))   # (N, 8, lookback)
# #         X_exo_full = np.transpose(data['X_exo'], (0, 2, 1))     # (N, 19, lookback)
# #         Y_target_full = np.transpose(data['Y_target'], (0, 2, 1)) # (N, 1, horizon)
        
# #         # FIX PER I DATI CONDIZIONALI (cond): Vettore statico, prendiamo l'ultimo frame
# #         X_cond_full = data['X_cond'][:, -1, :]                  # (N, 16) -> [Lati + Spawn = 16]
        
# #         # 3. SPLIT DATASET
# #         total_samples = len(X_endo_full)
# #         train_end = int(total_samples * train_ratio)
# #         val_end = int(total_samples * (train_ratio + val_ratio))
        
# #         if split == 'train':
# #             idx_start, idx_end = 0, train_end
# #         elif split == 'val':
# #             idx_start, idx_end = train_end, val_end
# #         elif split == 'test':
# #             idx_start, idx_end = val_end, total_samples
# #         else:
# #             raise ValueError("Lo split deve essere 'train', 'val' o 'test'.")
            
# #         self.X_endo = X_endo_full[idx_start:idx_end]
# #         self.X_exo = X_exo_full[idx_start:idx_end]
# #         self.X_cond = X_cond_full[idx_start:idx_end]
# #         self.Y_target = Y_target_full[idx_start:idx_end]
        
# #         # 4. GESTIONE GIOCATORI ASSENTI E VALORI DI SICUREZZA
# #         # L'estrattore ha usato -1.0 per segnare i sedili vuoti (Endo e Cond).
# #         # Per la rete neurale, l'assenza deve pesare zero (0.0).
# #         self.X_endo = np.where(self.X_endo == -1.0, 0.0, self.X_endo)
# #         self.Y_target = np.where(self.Y_target == -1.0, 0.0, self.Y_target)
# #         self.X_cond = np.where(self.X_cond == -1.0, 0.0, self.X_cond)
        
# #         # Pulizia di sicurezza per eventuali NaN sfuggiti da Pandas (diventano 0)
# #         self.X_exo = np.nan_to_num(self.X_exo, nan=0.0)

# #         # 5. TENSORIZZAZIONE
# #         self.X_endo = torch.tensor(self.X_endo, dtype=torch.float32)
# #         self.X_exo = torch.tensor(self.X_exo, dtype=torch.float32)
# #         self.X_cond = torch.tensor(self.X_cond, dtype=torch.float32)
# #         self.Y_target = torch.tensor(self.Y_target, dtype=torch.float32)

# #         print(f"[{split.upper()}] Creato con {len(self.X_endo)} campioni. (Norm: {self.norm})")

# #     def __len__(self):
# #         return len(self.X_endo)

# #     def __getitem__(self, idx):
# #         return self.X_endo[idx], self.X_exo[idx], self.X_cond[idx], self.Y_target[idx]


# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class VirtualAgentDataset(Dataset):
#     def __init__(self, npz_path, split='train', train_ratio=0.7, val_ratio=0.15, norm=None):
#         """
#         Carica e formatta il dataset del Virtual Agent ottimizzato per Big Data.
#         Utilizza operazioni In-Place e Zero-Copy per risparmiare decine di GB di RAM.
#         """
#         self.split = split
#         self.norm = norm
        
#         print(f"[{split.upper()}] Caricamento array dal disco in corso (Attendi qualche secondo)...")
#         # 1. Caricamento del Tensore Grezzo
#         data = np.load(npz_path)
        
#         # Troviamo il numero totale di campioni
#         total_samples = data['X_endo'].shape[0]
        
#         # 2. CALCOLO INDICI DI SPLIT
#         train_end = int(total_samples * train_ratio) #train_start = 0
#         val_end = int(total_samples * (train_ratio + val_ratio)) #val_start = train_end
#         #test_start = val_end; test_end = total_samples
        
#         if split == 'train':
#             idx_start, idx_end = 0, train_end
#         elif split == 'val':
#             idx_start, idx_end = train_end, val_end
#         elif split == 'test':
#             idx_start, idx_end = val_end, total_samples
#         else:
#             raise ValueError("Lo split deve essere 'train', 'val' o 'test'.")
            
#         print(f"[{split.upper()}] Estrazione della porzione {idx_start} -> {idx_end}...")
#         # Estraiamo SOLO la fetta che ci interessa prima di fare qualsiasi manipolazione!
#         X_endo_raw = data['X_endo'][idx_start:idx_end] 
#         X_exo_raw = data['X_exo'][idx_start:idx_end]
#         Y_target_raw = data['Y_target'][idx_start:idx_end]
#         X_cond_raw = data['X_cond'][idx_start:idx_end]
        
#         # Chiudiamo il file originale per svuotare il Garbage Collector di Python
#         data.close()

#         print(f"[{split.upper()}] Trasposizione assi...")
#         # 3. FIX DIMENSIONALE (Solo sulla porzione tagliata)
#         self.X_endo = np.transpose(X_endo_raw, (0, 2, 1))   # (N, 8, lookback)
#         self.X_exo = np.transpose(X_exo_raw, (0, 2, 1))     # (N, 19, lookback)
#         self.Y_target = np.transpose(Y_target_raw, (0, 2, 1)) # (N, 1, horizon)
#         self.X_cond = X_cond_raw[:, -1, :]                  # (N, 16)
        
#         print(f"[{split.upper()}] Pulizia memorie sporche (In-Place)...")
#         # 4. GESTIONE DI SICUREZZA 
#         # Usiamo copy=False per sovrascrivere direttamente in RAM senza raddoppiare l'uso della memoria
#         np.nan_to_num(self.X_exo, nan=0.0, copy=False) #copy=False evita di duplicare la memoria, sovrascrivendo direttamente l'array esistente

#         print(f"[{split.upper()}] Generazione Tensori Zero-Copy...")
#         # 5. TENSORIZZAZIONE (ZERO-COPY)
#         # from_numpy non duplica la memoria, crea solo un puntatore PyTorch ai dati Numpy esistenti!
#         self.X_endo = torch.from_numpy(self.X_endo).float()
#         self.X_exo = torch.from_numpy(self.X_exo).float()
#         self.X_cond = torch.from_numpy(self.X_cond).float()
#         self.Y_target = torch.from_numpy(self.Y_target).float()

#         print(f"[{split.upper()}] ✓ PRONTO! Creato con {len(self.X_endo)} campioni. (Endo={self.X_endo.shape[1]}, Exo={self.X_exo.shape[1]})")

#     def __len__(self):
#         return len(self.X_endo)

#     def __getitem__(self, idx):
#         return self.X_endo[idx], self.X_exo[idx], self.X_cond[idx], self.Y_target[idx]


import numpy as np
import torch
from torch.utils.data import Dataset

class VirtualAgentDataset(Dataset):
    def __init__(self, npz_path, split='train', norm=None):
        """
        Dataloader ottimizzato per il Virtual Agent.
        Legge direttamente i file già splittati per risparmiare RAM.
        """
        self.split = split
        self.norm = norm
        
        print(f"[{split.upper()}] Caricamento tensori da {npz_path} in corso...")
        
        # 1. Caricamento del Tensore (che ora è già tagliato per Train, Val o Test)
        data = np.load(npz_path)
        
        # 2. Trasposizione e Puntatori Zero-Copy
        # Usiamo torch.from_numpy e la trasposizione per non duplicare la RAM
        self.X_endo = torch.from_numpy(np.transpose(data['X_endo'], (0, 2, 1))).float()
        
        # Sostituiamo in-place gli eventuali NaN degli esogeni con 0.0 prima di tensorizzare
        exo_temp = data['X_exo']
        np.nan_to_num(exo_temp, nan=0.0, copy=False)
        self.X_exo = torch.from_numpy(np.transpose(exo_temp, (0, 2, 1))).float()
        
        # Cond: Vettore statico, prendiamo l'ultimo frame
        self.X_cond = torch.from_numpy(data['X_cond'][:, -1, :]).float()
        
        self.Y_target = torch.from_numpy(np.transpose(data['Y_target'], (0, 2, 1))).float()

        data.close()

        print(f"[{split.upper()}] ✓ PRONTO! (Campioni: {len(self.X_endo)})")

    def __len__(self):
        return len(self.X_endo)

    def __getitem__(self, idx):
        return self.X_endo[idx], self.X_exo[idx], self.X_cond[idx], self.Y_target[idx]