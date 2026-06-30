import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

def create_virtual_agent_dataset(csv_path, output_path, lookback=100, horizon=10, stride=0):
    print(f"\nLettura del CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X_endo_list, X_exo_list, X_cond_list, Y_target_list = [], [], [], []
    sessioni = df['ID_Sessione'].unique()
    NUM_PLAYERS = 8
    
    print("Estrazione delle finestre temporali (Self vs Others)...")
    for sessione in tqdm(sessioni, desc="Sessioni"):
        df_sess = df[df['ID_Sessione'] == sessione]
        tempi = np.sort(df_sess['Tempo'].unique())
        
        # 1. PIVOT: Colonne parallele per gli 8 posti
        pivot_dist = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='Dist3D').reindex(tempi).fillna(0.0)
        pivot_rowside = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='RowSide').reindex(tempi).fillna(0.0)
        
        # Zero-Padding per i posti vuoti
        for i in range(1, NUM_PLAYERS + 1):
            if i not in pivot_dist.columns:
                pivot_dist[i] = 0.0
                pivot_rowside[i] = 0.0
                
        # Dati ambientali condivisi
        primo_giocatore = df_sess['ID_Giocatore'].iloc[0]
        env_data = df_sess[df_sess['ID_Giocatore'] == primo_giocatore].set_index('Tempo').reindex(tempi).fillna(0.0)
        giocatori_reali = df_sess['ID_Giocatore'].unique()
        
        # 2. CREAZIONE DEI DATI "EGOCENTRICI"
        for player_id in giocatori_reali:
            
            # --- SEPARAZIONE SELF vs OTHERS ---
            self_dist = pivot_dist[player_id].values.reshape(-1, 1) 
            other_dists = pivot_dist.drop(columns=[player_id]).values[:, :7] 
            
            self_rowside = pivot_rowside[player_id].values.reshape(-1, 1)
            other_rowsides = pivot_rowside.drop(columns=[player_id]).values[:, :7]
            
            # --- ASSEMBLAGGIO (8 INPUTS) ---
            x_endo_full = np.column_stack([self_dist, other_dists]) 
            
            # --- EXO (18 INPUTS) ---
            x_exo_full = np.column_stack([
                env_data['SpeedLin_X'].values, env_data['SpeedLin_Z'].values,       
                env_data['SpeedAng_Y'].values,       
                env_data['Prua_X'].values, env_data['Prua_Z'].values,           
                env_data['Metronomo_Freq'].values,   
                env_data['Prop_Flag'].values, env_data['Prop_Z'].values,           
                env_data['RayFwd_X'].values, env_data['RayFwd_Z'].values,     
                env_data['RayLeft_X'].values, env_data['RayLeft_Z'].values,   
                env_data['RayRight_X'].values, env_data['RayRight_Z'].values, 
                env_data['Ray45_X'].values, env_data['Ray45_Z'].values,       
                env_data['Ray135_X'].values, env_data['Ray135_Z'].values      
            ])
            
            # --- COND (9 INPUTS) ---
            x_cond_full = np.column_stack([
                self_rowside,    
                other_rowsides,  
                env_data['TeamScore'].values
            ])
            
            # --- TARGET (1 OUTPUT) ---
            y_full = self_dist
            
            # --- SLIDING WINDOW ---
            num_samples = len(tempi)
            for t in range(0, num_samples - lookback - horizon, stride):
                X_endo_list.append(x_endo_full[t : t + lookback, :])
                X_exo_list.append(x_exo_full[t : t + lookback, :])
                X_cond_list.append(x_cond_full[t : t + lookback, :])
                Y_target_list.append(y_full[t + lookback : t + lookback + horizon, :])

    # 3. SALVATAGGIO
    print(f"\nSalvataggio nel file compresso: {output_path} in corso...")
    np.savez_compressed(
        output_path, 
        X_endo=np.array(X_endo_list, dtype=np.float32),
        X_exo=np.array(X_exo_list, dtype=np.float32),
        X_cond=np.array(X_cond_list, dtype=np.float32),
        Y_target=np.array(Y_target_list, dtype=np.float32)
    )
    
    print(f"\n--- RESOCONTO DATASET GENERATO (8 Input -> 1 Output) ---")
    print(f"X_endo  (Lookback Storico): {np.array(X_endo_list).shape} -> (Batch, Tempo, 8 Input: 1 Self + 7 Others)")
    print(f"Y_target(Orizzonte Futuro): {np.array(Y_target_list).shape} -> (Batch, {horizon} Frame, 1 Output: Self Futuro)")
    print("Elaborazione completata con successo!")

if __name__ == "__main__":
    # 1. Inizializza tkinter nascosto per il popup di selezione
    root = tk.Tk()
    root.withdraw()
    
    print("In attesa della selezione del file CSV...")
    percorso_csv = filedialog.askopenfilename(
        title="Seleziona il Dataset CSV generato da MATLAB",
        filetypes=[("CSV Files", "*.csv"), ("Tutti i file", "*.*")]
    )
    
    if percorso_csv:
        # Recuperiamo il percorso di esecuzione dello script
        cartella_script = os.path.dirname(os.path.abspath(__file__))
        
        # --- MODIFICATO: PUNTIAMO ESATTAMENTE ALLA CARTELLA PREPROCESS/VIRTUALAGENT ---
        cartella_output = os.path.join(cartella_script, "VirtualAgent")
        os.makedirs(cartella_output, exist_ok=True)
        
        percorso_output = os.path.join(cartella_output, "VA_Dataset_Tensor.npz")
        
        create_virtual_agent_dataset(
            csv_path=percorso_csv, 
            output_path=percorso_output,
            lookback=100, 
            horizon=10, 
            stride=5       
        )
    else:
        print("Nessun file selezionato. Operazione annullata.")