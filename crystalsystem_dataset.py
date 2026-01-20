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
import traceback
from utils.formula import ElementEncoder



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





def xrd_collate_fn(batch_data): 
    # 获取最大长度
    batch_size = len(batch_data)  # batch_data 是一个列表，包含多个样本
    peak_max_len = max([n['peaks_n'].item() for n in batch_data]) # 获取当前batch中peaks的最大长度。其中.item()将单元素张量转换为Python标量。
    formula_max_len = max([n['formula'].numel() for n in batch_data]) # 获取当前batch中formula的最大长度。
    
    # 对每个张量进行padding
    xrd_tensors = [] 
    filenames = []
    crystal_systems = []
    bandgaps = []
    formation_energies = []
    formulas = []
    lattice_parameters = []
    space_groups = []
    
    # 初始化batch的padding张量。创建全零的张量用于存放 padding 后的峰值、元素编码等。
    peaks_x_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_x'].dtype)
    peaks_y_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_y'].dtype)
    peaks_mask = torch.ones(batch_size, peak_max_len, dtype=torch.bool) # True表示padding位置，False表示有效数据位置。初始化为全True。
    
    formulas_padded_batch = torch.zeros(batch_size, formula_max_len, dtype=batch_data[0]['formula'].dtype)
    formulas_mask = torch.ones(batch_size, formula_max_len, dtype=torch.bool)
    
    # 遍历每个样本，填充到 batch 张量。把每个样本的 XRD、峰值、元素编码等填到对应的 batch 张量里。
    # 根据每个样本的实际长度，决定填充多少数据，剩余部分保持为零（padding）。（如 mask 标记为 True）
    for i, data in enumerate(batch_data):
        xrd_tensor = data['xrd_input']
        filename = data['xrd_file']
        peaks_x, peaks_y, peaks_n = data['peaks_x'], data['peaks_y'], data['peaks_n']

        xrd_tensors.append(xrd_tensor)
        filenames.append(filename)
        crystal_systems.append(data['crystal_system'])
        formation_energies.append(data['formation_energy'])
        bandgaps.append(data['bandgap'])
        lattice_parameters.append(data['lattice_parameter'])
        
        peak_seq_len = peaks_n.item()
        if peak_seq_len > 0:
            peaks_x_padded_batch[i, :peak_seq_len] = peaks_x[:peak_seq_len]
            peaks_y_padded_batch[i, :peak_seq_len] = peaks_y[:peak_seq_len]
            peaks_mask[i, :peak_seq_len] = False  # 标记有效数据位置为 False
            
        formulas_padded_batch[i, :len(data['formula'])] = data['formula']
        formulas_mask[i, :len(data['formula'])] = False
    
    xrd_input_batch = torch.stack(xrd_tensors, dim=0)
    crystal_systems_batch = torch.stack(crystal_systems, dim=0)
    formation_energies_batch = torch.stack(formation_energies, dim=0)
    lattice_parameters_batch = torch.stack(lattice_parameters, dim=0)
    
    return {
            'xrd_input': xrd_input_batch, 
            'xrd_filename': filenames, 
            'crystal_system': crystal_systems_batch,
            'peaks_x': peaks_x_padded_batch, 
            'peaks_y': peaks_y_padded_batch, 
            'peaks_mask': peaks_mask, 
            'formation_energy': formation_energies_batch,
            'bandgap': bandgaps,
            'formula': formulas_padded_batch,
            'formula_mask': formulas_mask,
            'lattice_parameter': lattice_parameters,
            }



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
        
        csdict = {"triclinic": 0,
            "monoclinic": 1,
            "orthorhombic": 2,
            "tetragonal": 3,
            "trigonal": 4,
            "hexagonal": 5,
            "cubic": 6}  # dictionary to map crystal string to float
        
        with open(json_path, 'r') as f:
            for line in f:
                # try:
                data = json.loads(line)
                filename = data['xrd']
                xrd_files.append(os.path.join(xrd_path, filename))
                # crystal_systems.append(data['crystal_system'])
                crystal_systems.append(csdict[data['crystal_system']])
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
    
    def __getitem__(self, idx): # 每个样本是字典，包含XRD、峰值、晶体系统等。返回是batch_data的元素。
        xrd_emb, peaks_x, peaks_y, peaks_n = self.get_xrd_embeddings(self.xrd_files[idx])
        crystal_system = torch.tensor(self.crystal_systems[idx], dtype=torch.float32)
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
                'crystal_system': crystal_system,
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



class XRDDataset(Dataset):
    def __init__(self,
                 xrd_path: str,
                 json_path: str,
                 label_to_extract="formation_energy"
                 ):
        print('start init dataset')
        xrd_files = []
        labels = []
        self.element_encoder = ElementEncoder() # 把函数放到这里，避免每次getitem都初始化。函数是对象属性的一部分。

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
                        filename = data['xrd']
                        xrd_files.append(os.path.join(xrd_path, filename))
                    elif label_to_extract == "crystal_system":
                        if not match:
                            print(f"未匹配到crystal_system: {data['conversations'][1]['value']}")
                            # 处理未匹配到的情况，报错终止
                            raise ValueError(f"未匹配到crystal_system: {data['conversations'][1]['value']}")
                        if match[0] not in csdict:
                            print(f"{match[0]} is not a legal crystal system!")
                            # 处理非法晶体系统，报错终止
                            raise ValueError(f"{match[0]} is not a legal crystal system!")
                        labels.append(csdict[match[0]]) # crystal string -> int
                        filename = data['xrd']
                        xrd_files.append(os.path.join(xrd_path, filename))
                except Exception as e:
                    print(f'Error skip line: {e}')
                    # 处理解析错误，报错终止                    
                    traceback.print_exc()

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