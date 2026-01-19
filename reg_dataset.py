import torch
import random
from typing import List, Optional, Tuple
import numpy as np
import os
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from scipy.signal import find_peaks
import re



def collate_fn(batch_data):
    # 获取最大长度
    batch_size = len(batch_data)
    peak_max_len = max([n['peaks_n'].item() for n in batch_data])
    # 对每个张量进行padding
    xrd_tensors = []
    filenames = []
    labels = []
    peaks_x_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_x'].dtype)
    peaks_y_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_y'].dtype)
    peaks_mask = torch.ones(batch_size, peak_max_len, dtype=torch.bool)
    for i, data in enumerate(batch_data):
        xrd_tensor = data['xrd_input']
        filename = data['xrd_file']
        peaks_x, peaks_y, peaks_n = data['peaks_x'], data['peaks_y'], data['peaks_n']

        xrd_tensors.append(xrd_tensor)
        filenames.append(filename)
        labels.append(data['label'])

        peak_seq_len = peaks_n.item()
        if peak_seq_len > 0:
            peaks_x_padded_batch[i, :peak_seq_len] = peaks_x[:peak_seq_len]
            peaks_y_padded_batch[i, :peak_seq_len] = peaks_y[:peak_seq_len]
            peaks_mask[i, :peak_seq_len] = False

    xrd_input_batch = torch.stack(xrd_tensors, dim=0)
    labels_batch = torch.stack(labels, dim=0)


    return {'xrd_input': xrd_input_batch, 'xrd_filename': filenames, 'peaks_x': peaks_x_padded_batch, 'peaks_y': peaks_y_padded_batch, 'peaks_mask': peaks_mask, 'labels': labels_batch}

def get_peaks(y):
    start = 5
    end = 80
    points_per_interval = 50
    # 计算总点数（75个区间，每个区间50个点，加上最后一个端点）
    total_points = (end - start) * points_per_interval
    angels = np.linspace(start, end, total_points) / 100
    peaks = find_peaks(y, height=0.01, prominence=0.02)[0]
    peak_x = angels[peaks]
    peak_y = y[peaks]
    peak_n = len(peak_x)
    return peaks, peak_x, peak_y, peak_n

class XRDDataset(Dataset):
    def __init__(self,
                 xrd_path: str,
                 json_path: str,
                 label_to_extract="formation_energy"
                 ):
        print('start init dataset')
        xrd_files = []
        labels = []

        if label_to_extract == "formation_energy":
            pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        elif label_to_extract == "crystal_system":
            pattern = r"belongs to (\w+) system"
            csdict = {"triclinic": 0,
                      "monoclinic": 1,
                      "orthorhombic": 2,
                      "tetragonal": 3,
                      "trigonal": 4,
                      "hexagonal": 5,
                      "cubic": 6}  # dictionary to map crystal string to float
        else:
            raise ValueError("label_to_extract should be 'formation_energy' or 'crystal_system'!")

        with open(json_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    match = re.findall(pattern, data["conversations"][1]["value"])
                    if label_to_extract == "formation_energy":
                        labels.append(float(match[0]))  # energy -> float
                    elif label_to_extract == "crystal_system":
                        if not match:
                            print(f"未匹配到晶系: {data["conversations"][1]["value"]}")
                            continue
                        elif match[0] not in csdict:
                            print(f"{match[0]} 不属于七种晶系: {data["conversations"][1]["value"]}")
                            continue
                        labels.append(csdict[match[0]]) # crystal string -> int
                    filename = data['xrd']
                    xrd_files.append(os.path.join(xrd_path, filename))
                except:
                    print('Error skip line')
                    continue

        self.xrd_files = np.array(xrd_files)    # list of strings -> np array of strings
        self.labels = np.array(labels)          # list of floats -> np array of floats
        print(f'finish init dataset, dataset size: {len(self.xrd_files)}')

    def __len__(self):
        return len(self.xrd_files)

    def __getitem__(self, idx):
        xrd_emb, peaks_x, peaks_y, peaks_n = self.get_xrd_embeddings(self.xrd_files[idx])
        label_idx = self.labels[idx]
        if isinstance(label_idx, int):
            label_idx = torch.tensor(label_idx, dtype=torch.int)
        else:
            label_idx = torch.tensor(label_idx, dtype=torch.float32)
        return {'xrd_input': xrd_emb, 'xrd_file': self.xrd_files[idx], 'peaks_x': peaks_x, 'peaks_y': peaks_y, 'peaks_n': peaks_n, 'label': label_idx}

    def get_xrd_embeddings(self, xrd_path):
        with open(xrd_path, 'r') as f:
            data = json.load(f)
        y = np.array(data['intensity'])
        y = (y - np.min(y)) / (np.max(y) - np.min(y))

        peaks, peaks_x, peaks_y, peaks_n = get_peaks(y)
        peaks_x = torch.tensor(np.array(peaks_x), dtype=torch.float32)
        peaks_y = torch.tensor(np.array(peaks_y), dtype=torch.float32)
        peaks_n = torch.tensor(peaks_n, dtype=torch.long)
        embeddings = torch.tensor(np.array(data['intensity']), dtype=torch.float32)

        return embeddings, peaks_x, peaks_y, peaks_n




if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from models.reg_model import XRDRegressionModel
    dataset = XRDDataset(
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path='/mnt/minio/battery/xrd/datasets/MP_bandgap-QA-test.jsonl'
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = XRDRegressionModel().to(torch.float32)
    for batch in train_loader:
        for key in batch:
            if key == 'xrd_filename':
                continue
            print(key, batch[key].shape)
        logits = model(batch['peaks_x'].to(torch.float32), batch['peaks_y'].to(torch.float32), batch['peaks_mask'])
        print(logits.shape)
        break