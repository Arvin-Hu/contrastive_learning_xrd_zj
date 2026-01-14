from trainer import ContrastiveLearningTrainer
from model import ContrastiveLearningModel
import torch
from dataset import ContrastiveLearningDataset, collate_fn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

if __name__ == "__main__":
    model = ContrastiveLearningModel(
        embedding_dim=256,
        projection_dim=64,
        use_qformer=True
    )
    model.load_state_dict(torch.load('/mnt/minio/battery/xrd/train_outputs/contrastive_learning/peak_v0/epoch_134.pth')['model_state_dict'])
    model.eval()  
    
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    
    train_dataset = ContrastiveLearningDataset(
        cif_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_cif_npy',
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path='/mnt/minio/battery/xrd/datasets/MP_bandgap-QA-train.jsonl'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False, collate_fn=collate_fn)
    
    batch = next(iter(train_loader))
    cif_input, xrd_input, cif_mask = batch['cif_input'], batch['xrd_input'], batch['cif_mask']
    
    dtype = next(model.parameters()).dtype
    progress_bar = tqdm(train_loader, desc=f'Export')
    emb_path = '/mnt/minio/battery/xrd/datasets/contrastive_learning/peaks_v0_epoch134/'
    if not os.path.exists(emb_path):
        os.system('mkdir -p {}'.format(os.path.join(emb_path,'xrd')))
        os.system('mkdir -p {}'.format(os.path.join(emb_path,'cif')))
    for batch in progress_bar:
        cif_input, cif_mask, peaks_x, peaks_y, peaks_mask = batch['cif_input'], batch['cif_mask'], batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
        cif_input = cif_input.to(device, dtype=dtype)
        cif_mask = cif_mask.to(device)
        peaks_x = peaks_x.to(device, dtype=dtype)
        peaks_y = peaks_y.to(device, dtype=dtype)
        peaks_mask = peaks_mask.to(device)
        
        filenames = batch['xrd_filename']
        
        _, xrd_emb, cif_emb = model(cif_input, cif_mask, peaks_x, peaks_y, peaks_mask)
        xrd_emb = xrd_emb.cpu().detach().numpy()
        cif_emb = cif_emb.cpu().detach().numpy()
        for i, filename in enumerate(filenames):
            xrd_out_file = os.path.join(emb_path, 'xrd', filename.split('/')[-1].replace('.json', '.npy'))
            if os.path.exists(xrd_out_file):
                continue
            np.save(xrd_out_file, xrd_emb[i])
            cif_out_file = os.path.join(emb_path, 'cif', filename.split('/')[-1].replace('.json', '.npy'))
            if os.path.exists(cif_out_file):
                continue
            np.save(cif_out_file, cif_emb[i])
    
    # progress_bar = tqdm(train_loader, desc=f'Export')
    # for batch in progress_bar:
    #     cif_input, xrd_input, cif_mask = batch['cif_input'], batch['xrd_input'], batch['cif_mask']
        
    #     cif_input = cif_input.to('cuda', dtype=dtype)
    #     xrd_input = xrd_input.to('cuda', dtype=dtype)
    #     cif_mask = cif_mask.to('cuda', dtype=dtype)
        
    #     filename = batch['xrd_filename']
    #     emb, _, _ = model(cif_input, xrd_input, cif_mask)
    #     emb = emb.cpu().numpy()