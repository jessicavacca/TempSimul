"""
===============================================================================
MODULE: UDP Telemetry & Loop-Back External Brain to Unity
===============================================================================
ARCHITECTURE ROLE:
This script acts as a lightweight UDP server connecting the Python backend 
to the Unity client. 

Its primary responsibilities are:
1. Health Monitoring: Calculates receive frequency (Hz) and detects dropped 
   packets in real-time to monitor the quality of the local/remote network. QUESTO VORREI MANTENERLO CI SERVE
2. Data Pass-Through (Loop-Back): Riceve la telemetria da Unity (68 byte), 
estrae le feature necessarie a nutrire il PatchTransformerForecast 
pre-addestrato (Dummy ECG con input_dim=8), esegue
l'inferenza in tempo reale sulla GPU e restituisce a Unity la coordinata Z 
predetta (4 byte) per controllare i movimenti dell'avatar.
===============================================================================
3. Command Handling: Listens for basic control strings like "PING" (for 
   connection checks) SERVE ANCORA- VORREI QUALCOSA CHE COMUNQUE MI DICA CHE STO RICEVENDO LE COSE
     and "QUIT" (for graceful shutdowns).

PAYLOAD EXPECTATION: CORREGGI (per ora mandiamo un ingresso di 8 float e un output di 1 float z coordinate)
- Control commands: 4 bytes (UTF-8 string)
- Telemetry data: 68 bytes (Binary C-struct: 1 uint, 1 long long, 14 floats)
===============================================================================
"""

import socket
import struct
import time
from collections import deque
import matplotlib
import numpy as np
import torch
import sys
import os
# Questo dice a Python di aggiungere la cartella principale del progetto
# alla lista dei posti dove cercare i moduli ("lib")
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Otteniamo la cartella ESATTA in cui si trova questo script (dentro il Docker)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assicuriamoci che Python sappia che 'script_dir' è il centro del mondo
sys.path.insert(0, script_dir)
from lib.Forecast.PatchTransformerExoForecast import PatchTransformerExoForecast #import the model architecture from the specified path
import matplotlib
matplotlib.use('Agg') # AGGIUNGI QUESTA RIGA: Forza la modalità background per non congelare il server
import matplotlib.pyplot as plt

# def salva_plot_predizione(input_tensor, prediction_tensor, current_id, channel=0):
#     """Salva un plot del passato e del futuro senza bloccare il server."""
#     # Estraiamo i dati dalla GPU/CPU ai numpy array normali
#     # input_tensor forma: (1, 8, 100). Prendiamo il canale desiderato (es. la Z).
#     lookback_data = input_tensor[0, channel, :].cpu().numpy()
    
#     # prediction_tensor forma: (1, 25). Prendiamo l'output futuro.
#     pred_data = prediction_tensor[0, :].cpu().numpy()

#     plt.figure(figsize=(12, 5))
    
#     # GRAFICO 1: I 100 frame storici (Passato)
#     plt.subplot(1, 2, 1)
#     plt.plot(lookback_data, label=f'Lookback (Canale {channel})', color='blue')
#     plt.title('Passato (Input da Unity)')
#     plt.grid(True)
#     plt.legend()
    
#     # GRAFICO 2: I 25 frame predetti (Futuro)
#     plt.subplot(1, 2, 2)
#     plt.plot(pred_data, label='Predizione Z', color='red', linestyle='--')
#     plt.title('Futuro (Output IA)')
#     plt.grid(True)
#     plt.legend()

#     # Salvataggio su file (sovrascrive sempre lo stesso file per non riempire il disco)
#     plt.savefig("debug_predizione_live.png")
#     plt.close() # FONDAMENTALE: chiude la figura per svuotare la RAM!

# --- CONFIGURATION ---
IP = "0.0.0.0"  # Listen on all available network interfaces
PORT = 65432    # Must exactly match the port used by the Unity client

# Initialize the socket
# AF_INET = IPv4 protocol | SOCK_DGRAM = UDP protocol
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
server_socket.bind((IP, PORT))  # Bind socket to the specified IP and port

# --- MODEL ARCHITECTURE CONFIGURATION ---
# These parameters must correspond exactly to the torchinfo configuration
LOOKBACK = 100
HORIZON = 25
INPUT_DIM = 8
PATCH_LEN = 10

# --- DATA UNPACKING FORMAT ---
# '<Iq14f' dictates how the binary data from C#/Unity is interpreted:
# '<'   : Little-Endian byte order (standard for most modern CPUs)
# 'I'   : unsigned int (4 bytes) -> Usually the Packet ID
# 'q'   : long long (8 bytes)    -> Usually a timestamp or high-res counter
# '14f' : 14 floats (14 * 4 = 56 bytes) -> Positional/Rotational data
# Total expected size = 68 bytes
BINARY_INPUT_FORMAT = '<Iq34f'
BINARY_OUTPUT_FORMAT = '<Iqf'

# --- HARDWARE INITIALISATION AND AI MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu") # For testing without GPU, comment out the line above and uncomment this one
print(f"[*] TELEMETRY Python Server listening on {IP}:{PORT}...")
print(f"[*] Hardware resource detected for AI inference: {device}")


# 1. Model instantiation with the same architecture used during training (must match exactly)
model = PatchTransformerExoForecast(
    lookback=100,        # Finestra storica
    horizon=25,          # Finestra di predizione
    input_dim=8,         # 8 canali cinematici in ingresso
    target_dim=1,        # 1 canale in uscita (la coordinata Z)
    condition_dim=8,     # Dimensione del vettore condizionale
    d_model=64,          # <-- DALLA TABELLA: La dimensione latente è 64
    n_heads=8,           # 8 teste di attenzione (standard per d_model=64)
    num_layers=2,        # 2 Transformer Encoder Layer
    norm_first=True,
    dropout=0,
    patch_len=10,
    layer_norm_eps=1e-5,
    bias=True,
    swiglu=True,
    rmsnorm=True,
    trans_norm=True,
    verbose=False
).to(device)

# 2. Loading the trained weights (PyTorch tutorial): weights_only=True to avoid compatibility issues with the device
#percorso_pesi = "weights_transformer_patch.pth"
# Trova la cartella esatta in cui risiede questo script (Notebooks)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Unisce il percorso della cartella al nome del file
percorso_pesi = os.path.join(script_dir, "weights_transformer_patch.pth")
try:
    model.load_state_dict(torch.load(percorso_pesi, map_location=device, weights_only=True))
    # FORZATURA ASSOLUTA: Spostiamo esplicitamente tutti i sottomoduli e buffer sulla CPU
    model = model.to(device)
    #model.load_state_dict(torch.load(percorso_pesi, map_location=device, weights_only=True), strict=False)
    print(f"[*] Weights of the model '{percorso_pesi}' loaded successfully on {device}!")
except FileNotFoundError:
    print(f"[!] WARNING: File '{percorso_pesi}' not found in the current directory.")
    print(f"[!] The model will perform inference with random weights (dummy initialization for stress testing).")

# 3. Impostiamo il modello in modalità valutazione (disattiva dropout)
model.eval()
torch.set_grad_enabled(False) # Disattiva i gradienti globalmente per risparmiare memoria e CPU
# 4. Inizializziamo la Sliding Window (Coda mobile) per la serie temporale
sliding_window = deque(maxlen=LOOKBACK)

print(f"[*] Server in ascolto su {IP}:{PORT}...")

# --- TELEMETRY TRACKERS ---
packets_received_sec = 0
total_packets_lost = 0
last_time = time.time()
last_received_id = 0

# --- MAIN SERVER LOOP ---
try:
    while True:
        # Buffer size is 2048 bytes (more than enough for 68-byte payloads)
        received_data, sender_address = server_socket.recvfrom(2048)
        data_length = len(received_data)

        # # 1. OPTIONAL SCREEN TELEMETRY UPDATE (Triggered once per second)
        # current_time = time.time()
        # if current_time - last_time >= 1.0:
        #     if packets_received_sec > 0:
        #         print(f"[NETWORK] Receive Rate: {packets_received_sec} Hz | Total Packet Loss: {total_packets_lost}")
        #     # Reset the per-second counter and update the timestamp
        #     packets_received_sec = 0
        #     last_time = current_time

        # 1. TEXT COMMAND HANDLING
        # If the payload is exactly 4 bytes, it might be a control string
        if data_length == 4:
            message = received_data.decode('utf-8').strip()          
            if message == "QUIT":
                print("\n[*] Clean shutdown requested by client.")
                break # Esce dal ciclo while e va al finally
            elif message == "PING":
                # Respond with PONG to verify the connection is alive
                server_socket.sendto("PONG".encode('utf-8'), sender_address)                
            continue # Skip the rest of the loop for text commands
            # Passa al prossimo pacchetto


        # 2. COORDINATE DATA HANDLING (148 bytes)
        elif data_length == 148:
            packets_received_sec += 1
            
            # Unpack the binary data into a Python tuple based on our format.
            # unpacked_data[0] is the 'I' (unsigned int), which represents the Packet ID
            unpacked_data = struct.unpack(BINARY_INPUT_FORMAT, received_data)
            current_id = unpacked_data[0]

            # 4. DROP RATE DETECTION
            # If the current ID skips a number compared to the last received ID, 
            # we know packets were lost in transit (common in UDP).
            if last_received_id > 0 and current_id > last_received_id + 1:
                total_packets_lost += (current_id - last_received_id - 1)
            
            last_received_id = current_id

            # ESTRAZIONE FEATURE (input_dim = 8):
            # unpacked_data[0] = Packet ID, unpacked_data[1] = Ticks temporali.
            # --- FASE DI TEST ---
            # I float cinematici partono dall'indice 2. Ne preleviamo esattamente 8.
            # Python ha letto 34 float, ma noi prendiamo solo i primi 8 (da indice 2 a 10)
            # per accontentare il modello ECG (input_dim=8).
            # I restanti 26 float vengono ignorati.
            current_features = np.array(unpacked_data[2:10], dtype=np.float32)
            
            sliding_window.append(current_features)
            
            # Aggiungiamo il vettore corrente alla memoria della sliding window
            sliding_window.append(current_features)
            
            # Calcolo della frequenza di ricezione (Hz) ogni secondo
            current_time = time.time()
            if current_time - last_time >= 1.0:
                print(f"[NETWORK] Ricezione: {packets_received_sec} Hz | Persi totali: {total_packets_lost} | Buffer storici: {len(sliding_window)}/{LOOKBACK}")
                packets_received_sec = 0
                last_time = current_time

            # Se la coda non è ancora piena (primi secondi di avvio), non possiamo fare inferenza.
            # Mandiamo un comando neutro (0.0) a Unity per non bloccare l'aggiornamento grafico.
            if len(sliding_window) < LOOKBACK:
                fallback_payload = struct.pack(BINARY_OUTPUT_FORMAT, current_id, unpacked_data[1], 0.0)
                server_socket.sendto(fallback_payload, sender_address)
                continue

            # PREDIZIONE REAL-TIME CON PATCH TRANSFORMER
            # 1. Convertiamo la deque in array numpy -> Forma: (100, 8)
            # 1. Matrice base (il lookback storico)
            # 1. Matrice base (il lookback storico)
            sequence_matrix = np.array(sliding_window) # Forma: (100, 8) 
            sequence_matrix_t = sequence_matrix.T      # Forma: (8, 100)
            
            # Creiamo il tensore PyTorch aggiungendo la dimensione del batch -> Forma: (1, 8, 100)
            input_tensor = torch.tensor(sequence_matrix_t, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 2. Generazione dei Dati Esogeni (Stress Test: Clone dell'input)
            exo_tensor = input_tensor.clone().to(device)

            # 3. Generazione dei Dati Condizionali (Stress Test: Estrazione dell'ultimo frame)
            cond_tensor = input_tensor[:, :, -1].clone().to(device) # Forma finale: (1, 8)

            # 4. Forward Pass con Multi-Input (TUTTO RIENTRATO NELL' ELIF)
            with torch.no_grad():
                # Passiamo tutti e tre i tensori alla rete contemporaneamente
                prediction = model(x=input_tensor, exo=exo_tensor, cond=cond_tensor) 
        
                # Preleviamo il primo valore futuro predetto (Z-Coordinate)
                predicted_z = prediction[0, 0].item()
                # print(f"Predizione Rete -> Z: {predicted_z:.4f}")
                
                # # --- NOVITÀ: SALVATAGGIO GRAFICO 1 VOLTA AL SECONDO ---
                # if current_id % 60 == 0:
                #     salva_plot_predizione(input_tensor, prediction, current_id, channel=0)
                #     print(f"[*] Grafico salvato per pacchetto ID {current_id}")
                # # ------------------------------------------------------
                
            # C. Invio del comando a Unity (ID, Ticks, Z) - Deve stare dentro l'ELIF!
            # ATTENZIONE: Assicurati che BINARY_OUTPUT_FORMAT sia '<Iqf' in cima al file
            output_payload = struct.pack(BINARY_OUTPUT_FORMAT, current_id, unpacked_data[1], predicted_z)
            server_socket.sendto(output_payload, sender_address)
            # 3. AGGIUNGI QUESTO BLOCCO ELSE:
        else:
            print(f"[!] ATTENZIONE: Ricevuto pacchetto ignorato di {data_length} byte. Mi aspettavo 148 byte o 4 byte.")
            
except KeyboardInterrupt:
    # QUESTO CATTURA IL TUO CTRL+C DAL TERMINALE
    print("\n[*] Interruzione manuale (Ctrl+C) rilevata. Spegnimento in corso...")

finally:
    # Cleanup when the loop breaks
    server_socket.close()
    print("[*] Socket UDP chiuso correttamente.")