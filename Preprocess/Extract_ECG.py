import numpy as np
import os
import argparse

# leads 'I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
# correlated leads 'III', 'aVR', 'aVL', 'aVF'

selected_leads = [0, 1, 6, 7, 8, 9, 10, 11]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract ECG data from .npz files and save as .npy')
    parser.add_argument('-i',
                        type=str,
                        required=True,
                        help='Directory containing input .npz files')
    parser.add_argument('-o',
                        type=str,
                        required=True,
                        help='Directory to save output .npy files')
    args = parser.parse_args()

    os.makedirs(args.o, exist_ok=True)

    for filename in os.listdir(args.i):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(args.i, filename))
            if 'arr_0' in data:
                data = data['arr_0']
                X_train = data[:, :, :-1]
                y_train = data[:, :, -1]
            else:
                try:
                    X_train = data['data']
                    y_train = data['labels']
                except:
                    X_train = data['samples']
                    y_train = data['classes']

            X_train = X_train[:, selected_leads]
            y_train = y_train[:, selected_leads]
            np.savez_compressed(
                os.path.join(args.o, filename.replace('.npz', '_data.npz')),
                X_train, y_train)
            print(f"Processed {filename}")
