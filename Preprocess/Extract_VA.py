import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def create_virtual_agent_dataset(csv_path, output_dir, lookback=100, horizon=25, stride=10):
    print(f"\nLettura del CSV: {csv_path}...")
    df = pd.read_csv(csv_path) #df contiene tutti i dati delle sessioni di gioco
    
    # =====================================================================
    # 1. NORMALIZZAZIONE GLOBALE PRE-ESTRAZIONE (Costanti Fisiche)
    # =====================================================================
    print("Calcolo delle metriche globali per la normalizzazione...")
    #df['Tempo'] = df['Tempo'].round(3)
    
    # A. Dati Endogeni (Distanza / Lunghezza Braccio)-> -> Range [0.0, 1.0] (Forzato con clip)
    df['Arm_Length'] = df['Arm_Length'].replace(0, np.nan) # Evitiamo divisioni per zero
    df['Dist3D'] = (df['Dist3D'] / df['Arm_Length']).clip(0.0, 1.0)
    #df['Dist3D'] = (df['Dist3D'] / df['Arm_Length'])

    
    # B. Dati Esogeni Semplici
    df['Metronomo_Freq'] = (df['Metronomo_Freq'] / 2.0).clip(0.0, 1.0)  # Max previsto: 2.0 Hz
    df['TeamScore'] = (df['TeamScore'] / 100.0).clip(0.0, 1.0)          # Max target: 100.0
    df['Boat_Angle'] = (df['Boat_Angle'] / 180.0).clip(-1.0, 1.0)        # Range: [-1.0, 1.0]

    # E. POSIZIONE BARCA (Prua_X, Prua_Z) nel mondo globale
    # Troviamo i confini massimi raggiunti dalla barca in tutto il dataset
    max_prua_x = df['Prua_X'].abs().max()
    max_prua_z = df['Prua_Z'].abs().max()
    # Evitiamo divisioni per zero se per caso la barca non si è mossa
    if pd.isna(max_prua_x) or max_prua_x == 0: max_prua_x = 1.0
    if pd.isna(max_prua_z) or max_prua_z == 0: max_prua_z = 1.0
    df['Prua_X'] = (df['Prua_X'] / max_prua_x).clip(-1.0, 1.0)
    df['Prua_Z'] = (df['Prua_Z'] / max_prua_z).clip(-1.0, 1.0)

    # C. Props (Costante Fisica Hardcoded)
    MAX_PROPS_DIST = 150.0
    for col in ['Distractor_dx', 'Distractor_dy', 'Distractor_dz', 'Attractor_dx', 'Attractor_dy', 'Attractor_dz']:
        df[col] = (df[col] / MAX_PROPS_DIST).clip(-1.0, 1.0)
        
    # D. Raycast (Costante Fisica Hardcoded)
    MAX_RAYCAST_DIST = 200.0
    ray_cols = ['RayFwd', 'RayLeft', 'RayRight', 'Ray45', 'Ray135']
    # Usiamo una ricerca flessibile per trovare tutte le colonne dei laser, a prescindere da maiuscole/minuscole
    ray_cols = [c for c in df.columns if c.startswith('Ray')]
    for col in ray_cols:
         df[col] = (df[col] / MAX_RAYCAST_DIST).clip(-1.0, 1.0)

    # F. Dati Condizionali (Statici)
    # Lo SpawnPoint va da 0 a 7, lo dividiamo per 7.0 per mapparlo esattamente in [0.0, 1.0]
    if 'SpawnPoint' in df.columns:
        df['SpawnPoint'] = (df['SpawnPoint'] / 7.0).clip(0.0, 1.0)
    
    # Il RowSide è già 0 o 1 (Sinistra o Destra). Lo blindiamo semplicemente in [0.0, 1.0]
    if 'RowSide' in df.columns:
        df['RowSide'] = df['RowSide'].clip(0.0, 1.0)

    # SALVATAGGIO DI DEBUG: Esporta il dataframe normalizzato per un controllo visivo
    check_path = os.path.join(output_dir, "Debug_Normalized_Data.csv")
    df.to_csv(check_path, index=False)
    print(f"File di check salvato in: {check_path}")


    # =====================================================================
    # 2. ESTRAZIONE E SPLIT SEQUENZIALE (Sessione per Sessione)
    # =====================================================================
    #X_endo_list, X_exo_list, X_cond_list, Y_target_list = [], [], [], []
    # Creiamo dizionari per raccogliere i dati già divisi!
    data_dict = {
        'train': {'endo': [], 'exo': [], 'cond': [], 'y': []},
        'val':   {'endo': [], 'exo': [], 'cond': [], 'y': []},
        'test':  {'endo': [], 'exo': [], 'cond': [], 'y': []}
    }
    sessioni = df['ID_Sessione'].unique()
    NUM_PLAYERS = 8 # La barca fisica ha sempre 8 posti
    
    print("Estrazione delle finestre temporali e Splitting per Sessione...")
    for sessione in tqdm(sessioni, desc="Sessioni"):
        df_sess = df[df['ID_Sessione'] == sessione]
        tempi = np.sort(df_sess['Tempo'].unique())
        
        # # 1. PIVOT: Creiamo una tabella con una colonna per ogni ID Giocatore presente
        # # [ACCORGIMENTO 1]: Uso -1.0 per Distanze, Rowside e SpawnPoint come segnale di "assenza" del giocatore
        # # 1. PIVOT SENZA FILLNA IMMEDIATO
        # pivot_dist = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='Dist3D').reindex(tempi)
        # pivot_rowside = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='RowSide').reindex(tempi)
        # pivot_spawn = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='SpawnPoint').reindex(tempi)
        
        # # --- PATCH: RAMMENDO DEI MICRO-BUCHI (PACKET DROP / JITTER) ---
        # # Interpoliamo i buchi piccoli fino a un massimo di 5 frame (es. mezzo secondo). 
        # # Poi, se il buco è più grande (vero abbandono), mettiamo -1.0

        # pivot_dist = pivot_dist.interpolate(method='linear', limit=5).fillna(-1.0)   
        # #absent_count = pivot_dist.isnull().sum()     
        # pivot_rowside = pivot_rowside.ffill(limit=5).fillna(-1.0)
        # pivot_spawn = pivot_spawn.ffill(limit=5).fillna(-1.0)
        
        # # 2. STANDARDIZZAZIONE A 8 SLOT FISSI (Zero Padding Sicuro)
        # # Costruiamo due nuove tabelle (Distanze e Lati) con esattamente 8 colonne (1 per sedile)
        # dist_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)
        # rowside_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)
        # spawn_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)

        # # Riempiamo gli slot vuoti con i dati dei giocatori reali che erano seduti lì
        # for col in pivot_dist.columns:
        #     if 1 <= col <= NUM_PLAYERS:
        #         dist_8_slots[col] = pivot_dist[col]
        #         rowside_8_slots[col] = pivot_rowside[col]
        #         spawn_8_slots[col] = pivot_spawn[col]
                
        # # Dati ambientali condivisi
        # env_data = df_sess.groupby('Tempo').first().reindex(tempi).interpolate(method='linear', limit=5)

        pivot_dist = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='Dist3D').reindex(tempi)
        
        # --- DEBUG TEMPORANEO (Da togliere dopo) ---
        if sessione == 1:
            print(f"\n--- DEBUG SESSIONE 1 ---")
            print(f"Giocatori trovati nel CSV: {df_sess['ID_Giocatore'].unique()}")
            print(f"NaN totali per giocatore prima dell'interpolazione:\n{pivot_dist.isnull().sum()}")
        # ------------------------------------------
        pivot_dist = pivot_dist.interpolate(method='linear', limit=30).fillna(-1.0)
        pivot_rowside = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='RowSide').reindex(tempi).ffill(limit=30).fillna(-1.0)
        pivot_spawn = df_sess.pivot(index='Tempo', columns='ID_Giocatore', values='SpawnPoint').reindex(tempi).ffill(limit=30).fillna(-1.0)
        
        dist_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)
        rowside_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)
        spawn_8_slots = pd.DataFrame(index=tempi, columns=range(1, NUM_PLAYERS + 1)).fillna(-1.0)

        for col in pivot_dist.columns:
            if 1 <= col <= NUM_PLAYERS:
                dist_8_slots[col] = pivot_dist[col]
                rowside_8_slots[col] = pivot_rowside[col]
                spawn_8_slots[col] = pivot_spawn[col]
                
        env_data = df_sess.groupby('Tempo').first().reindex(tempi).interpolate(method='linear', limit=5)

        # --- REGOLE DI FILLING POST-NORMALIZZAZIONE ---
        # --- REGOLE DI FILLING (Tutti i target mancanti vengono forzati ai loro valori di quiete fisici) ---
        # Avendo usato un match fuzzy sopra per i Raycast, andiamo a cercare i nomi esatti rimasti nel df
        rayFwd_Z_name = next((c for c in env_data.columns if c.lower() == 'rayfwd_dz' or c.lower() == 'rayfwd_z'), 'RayFwd_dz')
        rayLeft_Z_name = next((c for c in env_data.columns if c.lower() == 'rayleft_dz' or c.lower() == 'rayleft_z'), 'RayLeft_dz')
        rayRight_Z_name = next((c for c in env_data.columns if c.lower() == 'rayright_dz' or c.lower() == 'rayright_z'), 'RayRight_dz')
        ray45_Z_name = next((c for c in env_data.columns if c.lower() == 'ray45_dz' or c.lower() == 'ray45_z'), 'Ray45_dz')
        ray135_Z_name = next((c for c in env_data.columns if c.lower() == 'ray135_dz' or c.lower() == 'ray135_z'), 'Ray135_dz')
        
        rayFwd_X_name = next((c for c in env_data.columns if c.lower() == 'rayfwd_dx' or c.lower() == 'rayfwd_x'), 'RayFwd_dx')
        rayLeft_X_name = next((c for c in env_data.columns if c.lower() == 'rayleft_dx' or c.lower() == 'rayleft_x'), 'RayLeft_dx')
        rayRight_X_name = next((c for c in env_data.columns if c.lower() == 'rayright_dx' or c.lower() == 'rayright_x'), 'RayRight_dx')
        ray45_X_name = next((c for c in env_data.columns if c.lower() == 'ray45_dx' or c.lower() == 'ray45_x'), 'Ray45_dx')
        ray135_X_name = next((c for c in env_data.columns if c.lower() == 'ray135_dx' or c.lower() == 'ray135_x'), 'Ray135_dx')

        # _dz (Avanti) = 1.0 (Ostacolo lontanissimo)
        # _dx, _dy (Lati/Altezza) = 0.0 (Perfettamente centrato)
        fill_rules = {
            'Metronomo_Freq': 0.0,
            'TeamScore': 0.0, 
            'Distractor_dx': 0.0, 'Distractor_dy': 0.0, 'Distractor_dz': 1.0,
            'Attractor_dx': 0.0, 'Attractor_dy': 0.0, 'Attractor_dz': 1.0,
            rayFwd_X_name: 0.0, rayFwd_Z_name: 1.0,
            rayLeft_X_name: 0.0, rayLeft_Z_name: 1.0,
            rayRight_X_name: 0.0, rayRight_Z_name: 1.0,
            ray45_X_name: 0.0, ray45_Z_name: 1.0,
            ray135_X_name: 0.0, ray135_Z_name: 1.0
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
            self_spawn = spawn_8_slots[player_id].values.reshape(-1, 1)

            # Gli "Others" sono tutti gli altri 7 slot della barca (che siano umani o zeri)
            cols_others = [col for col in range(1, NUM_PLAYERS + 1) if col != player_id]
            other_dists = dist_8_slots[cols_others].values
            other_rowsides = rowside_8_slots[cols_others].values
            other_spawns = spawn_8_slots[cols_others].values

            # --- ASSEMBLAGGIO (8 INPUTS) ---
            # Colonna 0: Self. Colonne 1-7: Gli altri sedili (o zeri)
            x_endo_full = np.column_stack([self_dist, other_dists]) 
            
            # --- ASSEMBLAGGIO EXO (19 INPUTS) --- 
            # [ACCORGIMENTO 3]: TeamScore spostato qui dopo Metronomo_Freq 
            x_exo_full = np.column_stack([
                env_data['Prua_X'].values, env_data['Prua_Z'].values,           
                env_data['Boat_Angle'].values,                              
                env_data['Metronomo_Freq'].values,
                env_data['TeamScore'].values,   
                env_data['Distractor_dx'].values, env_data['Distractor_dz'].values,
                env_data['Attractor_dx'].values, env_data['Attractor_dz'].values,
                env_data[rayFwd_X_name].values, env_data[rayFwd_Z_name].values,     
                env_data[rayLeft_X_name].values, env_data[rayLeft_Z_name].values,   
                env_data[rayRight_X_name].values,  env_data[rayRight_Z_name].values, 
                env_data[ray45_X_name].values,  env_data[ray45_Z_name].values,       
                env_data[ray135_X_name].values, env_data[ray135_Z_name].values      
            ])
            
            # --- COND (16 INPUTS) ---
            x_cond_full = np.column_stack([
                self_rowside,    
                other_rowsides,  
                self_spawn,      
                other_spawns
            ])
            
            # --- TARGET (1 OUTPUT) ---
            y_full = self_dist

            # --- Liste temporanee per questo specifico giocatore in questa specifica sessione ---
            p_endo, p_exo, p_cond, p_y = [], [], [], []
            
            # --- SLIDING WINDOW ---
            num_samples = len(tempi)

            # --- INIZIO NUOVI CONTATORI ---
            finestre_generate = 0
            finestre_scartate = 0
            # --- FINE NUOVI CONTATORI ---

            for t in range(0, num_samples - lookback - horizon, stride):
                window_endo = x_endo_full[t : t + lookback, :]
                window_y = y_full[t + lookback : t + lookback + horizon, :]

                finestre_generate += 1 # Contiamo quante finestre "tenta" di creare
                # Se il giocatore 'Self' è assente (tutta la finestra è a -1.0), scartiamo il campione
                if np.any(window_endo[:, 0] == -1.0) or np.any(window_y == -1.0):
                    finestre_scartate += 1 # Contiamo quante finestre vengono scartate
                    continue
                # X_endo_list.append(x_endo_full[t : t + lookback, :])
                # X_exo_list.append(x_exo_full[t : t + lookback, :])
                # X_cond_list.append(x_cond_full[t : t + lookback, :])
                # Y_target_list.append(y_full[t + lookback : t + lookback + horizon, :])
                p_endo.append(window_endo)
                p_exo.append(x_exo_full[t : t + lookback, :])
                p_cond.append(x_cond_full[t : t + lookback, :])
                p_y.append(window_y)

            # [Opzionale] Stampa un mini-report per ogni giocatore di ogni sessione
            print(f"  -> Player {player_id}: generate {finestre_generate}, scartate {finestre_scartate} ({(finestre_scartate/finestre_generate)*100:.1f}%)")

    # 4. SALVATAGGIO
    # ULTIMO CLIP DI SICUREZZA PRE-SALVATAGGIO
    # Questo assicura matematicamente che la rete non vedrà MAI valori superiori a 1.0 (tranne i -1.0 degli assenti in Cond/Endo)
    # np_X_endo = np.clip(np.array(X_endo_list, dtype=np.float32), -1.0, 1.0)
    # np_X_exo = np.clip(np.array(X_exo_list, dtype=np.float32), -1.0, 1.0)
    # np_Y_target = np.clip(np.array(Y_target_list, dtype=np.float32), -1.0, 1.0)
    
#     np.savez_compressed(
#         output_path, 
#         X_endo=np_X_endo,
#         X_exo=np_X_exo,
#         X_cond=np.array(X_cond_list, dtype=np.float32),
#         Y_target=np_Y_target
#     )
    
#     print(f"\n--- RESOCONTO DATASET GENERATO ---")
#     print(f"X_endo  (Lookback Storico): {np.array(X_endo_list).shape} -> (Batch, Tempo, 8 Input: 1 Self + 7 Others)")
#     print(f"X_exo   (Esogeni Ambientali): {np_X_exo.shape} -> (Batch, Tempo, 19 Input)")
#     print(f"X_cond  (Statici): {np.array(X_cond_list).shape} -> (Batch, Tempo, 16 Input: Lati + Spawn)")
#     print(f"Y_target(Orizzonte Futuro): {np.array(Y_target_list).shape} -> (Batch, {horizon} Frame, 1 Output: Self Futuro)")
#     print("Elaborazione completata con successo! NESSUN DATO SUPERA I LIMITI FISICI.")


# if __name__ == "__main__":
#     # --- VERSIONE SERVER REMOTO (Punta direttamente alla cartella storage) ---
#     cartella_script = os.path.dirname(os.path.abspath(__file__))
    
#     # Poiché lo script è dentro 'Preprocess', usiamo '..' per salire di un livello alla root del progetto
#     # e poi entriamo in 'storage'
#     cartella_storage = os.path.abspath(os.path.join(cartella_script, "..", "storage"))
    
#     # Assicurati che il nome del CSV combaci con quello che hai caricato!
#     percorso_csv = os.path.join(cartella_storage, "Dataset_VirtualAgent_Clean_30Hz.csv")
#     percorso_output = os.path.join(cartella_storage, "VA_Dataset_Tensor.npz")
    
#     print(f"Cerco il file CSV in: {percorso_csv}")
    
#     if os.path.exists(percorso_csv):
#         create_virtual_agent_dataset(
#             csv_path=percorso_csv, 
#             output_path=percorso_output,
#             lookback=100,  #100 frame= circa 3.3 secondi a 30Hz
#             horizon=25,  # 25 frame (circa 0.83 secondi a 30Hz)
#             stride=90  # 90 frame (circa 3 secondi a 30Hz)      
#         )
#     else:
#         print(f"ERRORE: File CSV non trovato nel percorso {percorso_csv}!")

            # --- SPLIT 70-15-15 DELLA LINEA TEMPORALE DEL GIOCATORE ---
            n_tot = len(p_endo)
            if n_tot == 0: continue
            
            tr_end = int(n_tot * 0.7)
            val_end = int(n_tot * 0.85)
            
            data_dict['train']['endo'].extend(p_endo[:tr_end])
            data_dict['train']['exo'].extend(p_exo[:tr_end])
            data_dict['train']['cond'].extend(p_cond[:tr_end])
            data_dict['train']['y'].extend(p_y[:tr_end])
            
            data_dict['val']['endo'].extend(p_endo[tr_end:val_end])
            data_dict['val']['exo'].extend(p_exo[tr_end:val_end])
            data_dict['val']['cond'].extend(p_cond[tr_end:val_end])
            data_dict['val']['y'].extend(p_y[tr_end:val_end])
            
            data_dict['test']['endo'].extend(p_endo[val_end:])
            data_dict['test']['exo'].extend(p_exo[val_end:])
            data_dict['test']['cond'].extend(p_cond[val_end:])
            data_dict['test']['y'].extend(p_y[val_end:])

    # 4. SALVATAGGIO IN 3 FILE SEPARATI
    print(f"\nSalvataggio dei Tensori Divisi in corso...")
    
    for split_name in ['train', 'val', 'test']:
        out_file = os.path.join(output_dir, f"VA_Dataset_{split_name.capitalize()}.npz")
        
        np_X_endo = np.clip(np.array(data_dict[split_name]['endo'], dtype=np.float32), -1.0, 1.0)
        # se non volessi clippare np_X_endo, potrei commentare la riga sopra e usare direttamente:
        #np_X_endo = np.array(data_dict[split_name]['endo'], dtype=np.float32)
        np_X_exo = np.clip(np.array(data_dict[split_name]['exo'], dtype=np.float32), -1.0, 1.0)
        np_X_cond = np.clip(np.array(data_dict[split_name]['cond'], dtype=np.float32), -1.0, 1.0)
        np_Y_target = np.clip(np.array(data_dict[split_name]['y'], dtype=np.float32), -1.0, 1.0)
        #np_Y_target = np.array(data_dict[split_name]['y'], dtype=np.float32)

        
        np.savez_compressed(out_file, X_endo=np_X_endo, X_exo=np_X_exo, X_cond=np_X_cond, Y_target=np_Y_target)
        
        print(f"[{split_name.upper()}] Salvato: {len(np_X_endo)} campioni -> {out_file}")

    print("Elaborazione completata con successo! NESSUN DATO SUPERA I LIMITI FISICI.")

if __name__ == "__main__":
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    cartella_storage = os.path.abspath(os.path.join(cartella_script, "..", "storage"))
    percorso_csv = os.path.join(cartella_storage, "Dataset_VirtualAgent_Clean_30Hz.csv")
    
    if os.path.exists(percorso_csv):
        create_virtual_agent_dataset(
            csv_path=percorso_csv, 
            output_dir=cartella_storage, # Ora passiamo la CARTELLA, non il file
            lookback=100,  
            horizon=25,  
            stride=10       
        )
    else:
        print(f"ERRORE: File CSV non trovato nel percorso {percorso_csv}!")