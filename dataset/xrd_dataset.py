import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import torch
import random
from typing import List, Optional, Tuple
import numpy as np

from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from scipy.signal import find_peaks
import re
from utils.formula import ElementEncoder



def xrd_collate_fn(batch_data): 
    # 获取最大长度
    batch_size = len(batch_data)
    peak_max_len = max([n['peaks_n'].item() for n in batch_data])
    formula_max_len = max([n['formula'].numel() for n in batch_data])
    # 对每个张量进行padding
    xrd_tensors = []
    filenames = []
    crystal_systems = []
    bandgaps = []
    formation_energies = []
    formulas = []
    lattice_parameters = []
    space_groups = []
    
    peaks_x_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_x'].dtype)
    peaks_y_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_y'].dtype)
    peaks_mask = torch.ones(batch_size, peak_max_len, dtype=torch.bool)
    
    formulas_padded_batch = torch.zeros(batch_size, formula_max_len, dtype=batch_data[0]['formula'].dtype)
    formulas_mask = torch.ones(batch_size, formula_max_len, dtype=torch.bool)
    
    for i, data in enumerate(batch_data):
        xrd_tensor = data['xrd_input']
        filename = data['xrd_file']
        peaks_x, peaks_y, peaks_n = data['peaks_x'], data['peaks_y'], data['peaks_n']

        xrd_tensors.append(xrd_tensor)
        filenames.append(filename)
        formation_energies.append(data['formation_energy'])
        bandgaps.append(data['bandgap'])
        lattice_parameters.append(data['lattice_parameter'])
        
        peak_seq_len = peaks_n.item()
        if peak_seq_len > 0:
            peaks_x_padded_batch[i, :peak_seq_len] = peaks_x[:peak_seq_len]
            peaks_y_padded_batch[i, :peak_seq_len] = peaks_y[:peak_seq_len]
            peaks_mask[i, :peak_seq_len] = False
            
        formulas_padded_batch[i, :len(data['formula'])] = data['formula']
        formulas_mask[i, :len(data['formula'])] = False
    
    xrd_input_batch = torch.stack(xrd_tensors, dim=0)
    formation_energies_batch = torch.stack(formation_energies, dim=0)
    lattice_parameters_batch = torch.stack(lattice_parameters, dim=0)
    
    return {
            'xrd_input': xrd_input_batch, 
            'xrd_filename': filenames, 
            'peaks_x': peaks_x_padded_batch, 
            'peaks_y': peaks_y_padded_batch, 
            'peaks_mask': peaks_mask, 
            'formation_energy': formation_energies_batch,
            'bandgap': bandgaps,
            'formula': formulas_padded_batch,
            'formula_mask': formulas_mask,
            'lattice_parameter': lattice_parameters,
            }

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

class XRDFullDataset(Dataset):
    def __init__(self, 
                 xrd_path: str,
                 json_path: str,
                 ):
        print('start init dataset')
        xrd_files = []
        crystal_systems = []
        bandgaps = []
        formation_energies = []
        formulas = []
        lattice_parameters = []
        space_groups = []
        with open(json_path, 'r') as f:
            for line in f:
                # try:
                data = json.loads(line)
                filename = data['xrd']
                xrd_files.append(os.path.join(xrd_path, filename))
                crystal_systems.append(data['crystal_system'])
                bandgaps.append(data['bandgap'])
                formulas.append(data['formula'])
                lattice_parameters.append(data['lattice_parameters'])
                space_groups.append(data['space_group'])
                formation_energies.append(data['formation_energy'])
                # except:
                #     print('Error skip line')
                #     continue
        
        self.xrd_files = np.array(xrd_files)
        self.crystal_systems = np.array(crystal_systems)
        self.bandgaps = np.array(bandgaps)
        self.formulas = np.array(formulas)
        self.lattice_parameters = np.array(lattice_parameters)
        self.space_groups = np.array(space_groups)
        self.formation_energies = np.array(formation_energies)
        self.formula_encoder = ElementEncoder()
        print('finish init dataset, dataset size: {}'.format(len(self.xrd_files)))
        
    def __len__(self):
        return len(self.xrd_files)
    
    def __getitem__(self, idx):
        xrd_emb, peaks_x, peaks_y, peaks_n = self.get_xrd_embeddings(self.xrd_files[idx])
        # crystal_system = torch.tensor(self.crystal_systems[idx], dtype=torch.float32)
        bandgap = torch.tensor(self.bandgaps[idx], dtype=torch.float32)
        encoded = self.formula_encoder.encode_formula(self.formulas[idx])
        formula = torch.tensor(encoded, dtype=torch.int64)
        lattice_parameter = torch.tensor(self.lattice_parameters[idx], dtype=torch.float32)
        # space_group = torch.tensor(self.space_groups[idx], dtype=torch.float32)
        formation_energy = torch.tensor(self.formation_energies[idx], dtype=torch.float32)
        return {'xrd_input': xrd_emb, 
                'xrd_file': self.xrd_files[idx],
                'peaks_x': peaks_x, 
                'peaks_y': peaks_y, 
                'peaks_n': peaks_n, 
                'formation_energy': formation_energy, 
                # 'crystal_system': crystal_system,
                'formula': formula,
                'lattice_parameter': lattice_parameter,
                # 'space_group': space_group,
                'bandgap': bandgap
                }
    
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
    from models.reg_model import XRDRegressionModel, XRDFormulaModel
    dataset = XRDFullDataset(
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path='/mnt/minio/battery/xrd/datasets/MP_xrd-test.jsonl'
    )
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=xrd_collate_fn)
    model = XRDFormulaModel().to(torch.float32)
    for batch in train_loader:
        for key in batch:
            if key == 'xrd_filename':
                continue
            print(key, batch[key])
        logits = model(batch['peaks_x'].to(torch.float32), batch['peaks_y'].to(torch.float32), batch['peaks_mask'], batch['formula'].to(torch.int64), batch['formula_mask'])
        print(logits.shape)
        break