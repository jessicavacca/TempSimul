import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract ECG data from .npz files and save as .npy')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output .npy files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.npz'):
            data = np.load(os.path.join(args.input_dir, filename))
            X_train = data['data']
            np.save(os.path.join(args.output_dir, filename.replace('.npz', '_data.npy')), X_train)
            print(f"Processed {filename}")