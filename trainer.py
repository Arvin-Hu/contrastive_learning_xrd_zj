import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
from models.model import ContrastiveLoss
from sklearn.metrics import r2_score
from tqdm import tqdm
import os

import gc

def debug_memory():
    print("="*50)
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    print(f"Max allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    
    # 检查哪些张量占用内存
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    size_mb = obj.element_size() * obj.nelement() / 1024**3
                    if size_mb > 0.1:  # 大于100MB
                        print(f"Large tensor: {type(obj).__name__}, size: {size_mb:.2f}GB, shape: {obj.shape}")
        except:
            pass
    print("="*50)
    
    

class ContrastiveLearningTrainer:
    """
    对比学习训练器
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-4,
                 temperature: float = 0.1,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_fn = ContrastiveLoss()
        self.temperature = temperature
        self.dtype=next(self.model.parameters()).dtype
        self.start_epoch = 0
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 50,  # 假设训练50个epoch
            eta_min=1e-6
        )
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
    
    def train_epoch(self, epoch) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress_bar:
            cif_input, cif_mask, peaks_x, peaks_y, peaks_mask = batch['cif_input'], batch['cif_mask'], batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
            
            cif_input = cif_input.to(self.device, dtype=self.dtype)
            cif_mask = cif_mask.to(self.device)
            peaks_x = peaks_x.to(self.device, dtype=self.dtype)
            peaks_y = peaks_y.to(self.device, dtype=self.dtype)
            peaks_mask = peaks_mask.to(self.device)
            
            # 前向传播
            _, xrd_proj, cif_proj = self.model(cif_input, cif_mask, peaks_x, peaks_y, peaks_mask)
            
            # 计算对比损失
            loss = self.loss_fn.infonce_loss(xrd_proj, cif_proj, temperature=self.temperature)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_count += 1
            total_loss += loss.item()
                    
        avg_loss = total_loss / batch_count
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """验证"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        
        for batch in self.val_loader:
            cif_input, cif_mask, peaks_x, peaks_y, peaks_mask = batch['cif_input'], batch['cif_mask'], batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
            
            cif_input = cif_input.to(self.device, dtype=self.dtype)
            cif_mask = cif_mask.to(self.device)
            peaks_x = peaks_x.to(self.device, dtype=self.dtype)
            peaks_y = peaks_y.to(self.device, dtype=self.dtype)
            peaks_mask = peaks_mask.to(self.device)
            
            # 前向传播
            _, xrd_proj, cif_proj = self.model(cif_input, cif_mask, peaks_x, peaks_y, peaks_mask)
            
            loss = self.loss_fn.infonce_loss(xrd_proj, cif_proj, temperature=self.temperature)
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs: int, save_path: Optional[str] = None):
        """训练循环"""
        print("开始训练...")
        
        for epoch in range(self.start_epoch, num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印进度
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存模型
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.save_model(os.path.join(save_path, f"epoch_{epoch+1}.pth"), epoch)
        
        if save_path:
            self.save_model(os.path.join(save_path, f"final.pth"), epoch)
        
        print("训练完成！")
    
    def save_model(self, path: str, epoch: int=0):
        """保存模型"""
        info = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'epoch': epoch,
        }
        torch.save(info, path)
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.start_epoch = checkpoint['epoch']
        print('模型加载: ', path)        
        


class RegressionTrainer(ContrastiveLearningTrainer):
    def train_epoch(self, epoch) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress_bar:
            peaks_x, peaks_y, peaks_mask = batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
            labels = batch['labels']
            peaks_x = peaks_x.to(self.device, dtype=self.dtype)
            peaks_y = peaks_y.to(self.device, dtype=self.dtype)
            peaks_mask = peaks_mask.to(self.device)
            labels = labels.to(self.device, dtype=self.dtype)
            
            # 前向传播
            logits = self.model(peaks_x, peaks_y, peaks_mask)
            
            # 计算loss
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits.squeeze(1), labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_count += 1
            total_loss += loss.item()
                    
        avg_loss = total_loss / batch_count
        self.history['train_loss'].append(avg_loss)
        self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> Optional[float]:
        """验证"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        batch_count = 0
        label_list, pred_list = [], []
        for batch in self.val_loader:
            peaks_x, peaks_y, peaks_mask = batch['peaks_x'], batch['peaks_y'], batch['peaks_mask']
            labels = batch['labels']
            peaks_x = peaks_x.to(self.device, dtype=self.dtype)
            peaks_y = peaks_y.to(self.device, dtype=self.dtype)
            peaks_mask = peaks_mask.to(self.device)
            labels = labels.to(self.device, dtype=self.dtype)
            
            # 前向传播
            logits = self.model(peaks_x, peaks_y, peaks_mask)
            
            # 计算loss
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(logits.squeeze(1), labels)
            
            total_loss += loss.item()
            batch_count += 1
            
            logits = logits.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            
            label_list.extend(labels.reshape(-1).tolist())
            pred_list.extend(logits.reshape(-1).tolist())
            
        
        r2 = r2_score(np.array(label_list), np.array(pred_list))
        mae = np.abs(np.array(label_list) - np.array(pred_list)).mean()
        print('r2: {}, mae: {}'.format(r2, mae))
        
        avg_loss = total_loss / batch_count
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss