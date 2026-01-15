import numpy as np
import os
from tqdm import tqdm
import sys
from models.reg_model import XRDRegressionModel
from reg_dataset import XRDDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

if __name__ == '__main__':
    filename = sys.argv[1] # /mnt/minio/battery/xrd/train_outputs/xrd/formation_energy/final.pth
    print(filename)
    model = XRDRegressionModel(
        embedding_dim=256,
    )
    model.load_state_dict(torch.load(filename)['model_state_dict'])
    model.eval()  
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    dtype = next(model.parameters()).dtype
    
    dataset = XRDDataset(
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path='/mnt/minio/battery/xrd/datasets/MP_formationenergy-QA-train.jsonl'
    )
    
    eval_loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)
    progress_bar = tqdm(eval_loader, desc=f'Eval')
    label_list, pred_list = [], []
    
    for batch in progress_bar:
        peaks_x, peaks_y, peaks_mask = batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
        labels = batch['labels']
        peaks_x = peaks_x.to(device, dtype=dtype)
        peaks_y = peaks_y.to(device, dtype=dtype)
        peaks_mask = peaks_mask.to(device)
        labels = labels.to(device, dtype=dtype)
        
        # 前向传播
        logits = model(peaks_x, peaks_y, peaks_mask)
        
        logits = logits.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        label_list.extend(labels.reshape(-1).tolist())
        pred_list.extend(logits.reshape(-1).tolist())
        
    
    r2 = r2_score(np.array(label_list), np.array(pred_list))
    mae = np.abs(np.array(label_list) - np.array(pred_list)).mean()
    print('r2: {}, mae: {}'.format(r2, mae))