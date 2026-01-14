import torch
import random
from typing import List, Optional, Tuple
import numpy as np
import os
from torch.utils.data import Dataset
import json
import torch.nn.functional as F
from chgnet.model import CHGNet
from pymatgen.core import Structure
from scipy.signal import find_peaks

class DataAugmentation:
    """
    数据增强策略（根据具体任务定制）
    """
    @staticmethod
    def random_mask(x: torch.Tensor, mask_ratio: float = 0.2) -> torch.Tensor:
        """随机掩码增强"""
        batch_size, seq_len = x.shape
        mask = torch.rand(x.shape) > mask_ratio
        return x * mask.float()
    
    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """高斯噪声增强"""
        noise = torch.randn_like(x) * std
        return x + noise
    
    @staticmethod
    def random_shuffle(x: torch.Tensor) -> torch.Tensor:
        """随机打乱序列"""
        idx = torch.randperm(x.size(1))
        return x[:, idx]
    





def collate_fn(batch_data): 
    # 获取最大长度
    cif_max_length = max(data['cif_input'].shape[0] for data in batch_data)
    batch_size = len(batch_data)
    peak_max_len = max([n['peaks_n'].item() for n in batch_data])
    # 对每个张量进行padding
    cif_mask = torch.zeros(batch_size, cif_max_length, dtype=torch.int16)
    xrd_tensors = []
    padded_tensors = []
    filenames = []
    peaks_x_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_x'].dtype)
    peaks_y_padded_batch = torch.zeros(batch_size, peak_max_len, dtype=batch_data[0]['peaks_y'].dtype)
    peaks_mask = torch.ones(batch_size, peak_max_len, dtype=torch.bool)
    for i, data in enumerate(batch_data):
        cif_tensor = data['cif_input']
        xrd_tensor = data['xrd_input']
        filename = data['xrd_file']
        peaks_x, peaks_y, peaks_n = data['peaks_x'], data['peaks_y'], data['peaks_n']
        cif_seq_len, output_dim = cif_tensor.shape
        pad_size = cif_max_length - cif_seq_len
        
        if pad_size > 0:
            # 增加batch维度并padding
            cif_tensor = cif_tensor.unsqueeze(0)  # [1, seq_len, output_dim]
            pad_shape = (0, 0, 0, pad_size, 0, 0)
            padded = F.pad(cif_tensor, pad_shape, value=0).squeeze(0)
        else:
            padded = cif_tensor[:cif_max_length, :]
        
        padded_tensors.append(padded)   
        xrd_tensors.append(xrd_tensor)
        cif_mask[i, :cif_seq_len] = 1
        filenames.append(filename)
        
        peak_seq_len = peaks_n.item()
        if peak_seq_len > 0:
            peaks_x_padded_batch[i, :peak_seq_len] = peaks_x[:peak_seq_len]
            peaks_y_padded_batch[i, :peak_seq_len] = peaks_y[:peak_seq_len]
            peaks_mask[i, :peak_seq_len] = False
    
    cif_input_batch = torch.stack(padded_tensors, dim=0)
    xrd_input_batch = torch.stack(xrd_tensors, dim=0)
    
    return {'cif_input': cif_input_batch, 'xrd_input': xrd_input_batch, 'cif_mask': cif_mask, 'xrd_filename': filenames, 'peaks_x': peaks_x_padded_batch, 'peaks_y': peaks_y_padded_batch, 'peaks_mask': peaks_mask}

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

class ContrastiveLearningDataset(Dataset):
    """
    对比学习数据集
    """
    def __init__(self, 
                 cif_path: str, 
                 xrd_path: str,
                 json_path: str,
                 augmentations: Optional[List] = None):
        print('start init dataset')
        xrd_files = []
        cif_files = []
        with open(json_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                filename = data['xrd']
                xrd_files.append(os.path.join(xrd_path, filename))
                cif_files.append(os.path.join(cif_path, filename.replace('.json', '.npy')))
        
        assert len(cif_files) == len(xrd_files)
        
        self.xrd_files = np.array(xrd_files)
        self.cif_files = np.array(cif_files)

        self.augmentations = augmentations or []
        # self.model = CHGNet.load()
        print('finish init dataset')
        
    def __len__(self):
        return len(self.cif_files)
    
    def __getitem__(self, idx):
        cif_emb = self.get_atom_embeddings_with_chgnet(self.cif_files[idx])
        xrd_emb, peaks_x, peaks_y, peaks_n = self.get_xrd_embeddings(self.xrd_files[idx])
        
        return {'cif_input': cif_emb, 'xrd_input': xrd_emb, 'xrd_file': self.xrd_files[idx], 'peaks_x': peaks_x, 'peaks_y': peaks_y, 'peaks_n': peaks_n}
    
    def get_atom_embeddings_with_chgnet(self, cifpaths):
        return torch.tensor(np.load(cifpaths), dtype=torch.float32)
    
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
    
    # def _apply_augmentations(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """应用数据增强"""
    #     x1 = x.clone()
    #     x2 = x.clone()
        
    #     for aug in self.augmentations:
    #         if random.random() > 0.5:
    #             x1 = aug(x1)
    #         if random.random() > 0.5:
    #             x2 = aug(x2)
        
    #     return x1, x2
    
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from model import ContrastiveLearningModel
    dataset = ContrastiveLearningDataset(
        cif_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_cif_npy',
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path='/mnt/minio/battery/xrd/datasets/MP_bandgap-QA-train.jsonl'
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model = ContrastiveLearningModel(
        xrd_input_dim=3751,
    ).to(torch.float32)
    for batch in train_loader:
        for key in batch:
            if key == 'xrd_filename':
                continue
            print(key, batch[key].shape)
        xrd_emb, xrd_proj, cif_proj = model(batch['cif_input'].to(torch.float32), batch['xrd_input'].to(torch.float32), batch['cif_mask'])
        print(xrd_emb.shape, xrd_proj.shape, cif_proj.shape)
        break