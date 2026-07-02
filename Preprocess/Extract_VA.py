import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def create_virtual_agent_dataset(csv_path, output_path, lookback=100, horizon=25, stride=10):
    print(f"\nLettura del CSV: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    X_endo_list, X_exo_list, X_cond_list, Y_target_list = [], [], [], []
    sessioni = df['ID_Sessione'].unique()
    NUM_PLAYERS = 8 # La barca fisica ha sempre 8 posti
    
    print("Estrazione delle finestre temporali (Mappatura a Slot Fissi)...")
    for sessione in tqdm(sessioni, desc="Sessioni"):
        df_sess = df[df['ID_Sessione'] == sessione]
        tempi = np.sort(df_sess['Tempo'].unique())
        
        # 1. PIVOT: Creiamo una tabella con una colonna per ogni ID Giocatore presente
        pivot_dist = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='Dist3D').reindex(tempi).fillna(-1.0)
        pivot_rowside = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='RowSide').reindex(tempi).fillna(0.0)
        
        # 2. STANDARDIZZAZIONE A 8 SLOT FISSI (Zero Padding Sicuro)
        # Costruiamo due nuove tabelle (Distanze e Lati) con esattamente 8 colonne (1 per sedile)
        dist_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)
        rowside_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(0.0)
        
        # Riempiamo gli slot vuoti con i dati dei giocatori reali che erano seduti lì
        for col in pivot_dist.columns:
            if 1 <= col <= NUM_PLAYERS:
                dist_8_slots[col] = pivot_dist[col]
                rowside_8_slots[col] = pivot_rowside[col]
                
        # Dati ambientali condivisi
        primo_giocatore = df_sess['ID_Giocatore'].iloc[0]
        env_data = df_sess[df_sess['ID_Giocatore'] == primo_giocatore].set_index('Tempo').reindex(tempi)
        
        # --- APPLICHIAMO LE TUE REGOLE DI FILLING (Data Cleaning) ---
        # Evitiamo che il .fillna(0.0) globale distrugga la logica fisica dei dati!
        fill_rules = {
            'Metronomo_Freq': 0.0,
            'TeamScore': 30.0,
            'Distractor_dx': 999.0, 'Distractor_dy': 999.0, 'Distractor_dz': 999.0,
            'Attractor_dx': 999.0, 'Attractor_dy': 999.0, 'Attractor_dz': 999.0,
            'RayFwd_X': 999.0, 'RayFwd_Y': 999.0, 'RayFwd_Z': 999.0,
            'RayLeft_X': 999.0, 'RayLeft_Y': 999.0, 'RayLeft_Z': 999.0,
            'RayRight_X': 999.0, 'RayRight_Y': 999.0, 'RayRight_Z': 999.0,
            'Ray45_X': 999.0, 'Ray45_Y': 999.0, 'Ray45_Z': 999.0,
            'Ray135_X': 999.0, 'Ray135_Y': 999.0, 'Ray135_Z': 999.0
        }
        # Applichiamo il dizionario, e solo per le colonne restanti (es. Boat_Angle) mettiamo 0.0
        env_data = env_data.fillna(value=fill_rules).fillna(0.0)
        
        # Giocatori UMANI presenti in questa specifica sessione
        giocatori_reali = df_sess['ID_Giocatore'].unique()
        
        # 3. CREAZIONE DEI DATI "EGOCENTRICI"
        # Cicliamo SOLO sui giocatori veri per usarli come "Self"
        for player_id in giocatori_reali:
            
            # --- SEPARAZIONE SELF vs OTHERS (Usando gli 8 slot fissi) ---
            # Il Self è il giocatore corrente
            self_dist = dist_8_slots[player_id].values.reshape(-1, 1) 
            self_rowside = rowside_8_slots[player_id].values.reshape(-1, 1)
            
            # Gli "Others" sono tutti gli altri 7 slot della barca (che siano umani o zeri)
            cols_others = [col for col in range(1, NUM_PLAYERS + 1) if col != player_id]
            other_dists = dist_8_slots[cols_others].values
            other_rowsides = rowside_8_slots[cols_others].values
            
            # --- ASSEMBLAGGIO (8 INPUTS) ---
            # Colonna 0: Self. Colonne 1-7: Gli altri sedili (o zeri)
            x_endo_full = np.column_stack([self_dist, other_dists]) 
            
            # --- EXO (18 INPUTS) ---
            x_exo_full = np.column_stack([
                env_data['Prua_X'].values, env_data['Prua_Z'].values,           
                env_data['Boat_Angle'].values,                              
                env_data['Metronomo_Freq'].values,   
                env_data['Distractor_dx'].values, env_data['Distractor_dz'].values,
                env_data['Attractor_dx'].values, env_data['Attractor_dz'].values,
                env_data['RayFwd_X'].values, env_data['RayFwd_Z'].values,     
                env_data['RayLeft_X'].values, env_data['RayLeft_Z'].values,   
                env_data['RayRight_X'].values,  env_data['RayRight_Z'].values, 
                env_data['Ray45_X'].values,  env_data['Ray45_Z'].values,       
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
                window_endo = x_endo_full[t : t + lookback, :]
                window_y = y_full[t + lookback : t + lookback + horizon, :]
                
                # Se il giocatore 'Self' è assente (tutta la finestra è a -1.0), scartiamo il campione
                if np.any(window_endo[:, 0] == -1.0) or np.any(window_y == -1.0):
                    continue
                X_endo_list.append(x_endo_full[t : t + lookback, :])
                X_exo_list.append(x_exo_full[t : t + lookback, :])
                X_cond_list.append(x_cond_full[t : t + lookback, :])
                Y_target_list.append(y_full[t + lookback : t + lookback + horizon, :])

    # 4. SALVATAGGIO
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
    print(f"X_exo   (Esogeni Ambientali): {np.array(X_exo_list).shape} -> (Batch, Tempo, 18 Input)")
    print(f"Y_target(Orizzonte Futuro): {np.array(Y_target_list).shape} -> (Batch, {horizon} Frame, 1 Output: Self Futuro)")
    print("Elaborazione completata con successo!")

if __name__ == "__main__":
    # --- VERSIONE SERVER REMOTO (Punta direttamente alla cartella storage) ---
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    
    # Poiché lo script è dentro 'Preprocess', usiamo '..' per salire di un livello alla root del progetto
    # e poi entriamo in 'storage'
    cartella_storage = os.path.abspath(os.path.join(cartella_script, "..", "storage"))
    
    # Assicurati che il nome del CSV combaci con quello che hai caricato!
    percorso_csv = os.path.join(cartella_storage, "Dataset_VirtualAgent_Clean_MultiSession.csv")
    percorso_output = os.path.join(cartella_storage, "VA_Dataset_Tensor.npz")
    
    print(f"Cerco il file CSV in: {percorso_csv}")
    
    if os.path.exists(percorso_csv):
        create_virtual_agent_dataset(
            csv_path=percorso_csv, 
            output_path=percorso_output,
            lookback=100, 
            horizon=25,
            stride=10       
        )
    else:
        print(f"ERRORE: File CSV non trovato nel percorso {percorso_csv}!")