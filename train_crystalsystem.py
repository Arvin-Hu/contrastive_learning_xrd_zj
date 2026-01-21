import numpy as np
from models.crystalsystem_model import XRDClassificationModel,XRDFormulaClassificationModel
from models.reg_model import XRDRegressionModel, XRDConvRegressionModel, XRDFormulaModel
from trainer import (
    FormationEnergyTrainer,
    RegressionTrainer,
    CrystalSystemClassificationTrainer
)

from dataset.crystalsystem_dataset import XRDFullDataset, xrd_collate_fn
import torch
from torch.utils.data import DataLoader

def train(
    output_path,
    batch_size=128, 
    learning_rate=1e-4, 
    weight_decay=1e-4, 
    embedding_dim=256, 
    epochs=30,
    num_heads=8,
    num_layers=6,
    model_path=None,
    train_path=None,
    eval_path=None,
    log_dir="logs",
    model_class='XRDFormulaModel',
    trainer_class='RegressionTrainer'
):
    
    # 3. 创建数据集和数据加载器
    train_dataset = XRDFullDataset(
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path=train_path
    )
    eval_dataset = XRDFullDataset(
        xrd_path='/mnt/minio/battery/xrd/datasets/raw_data/mp_xrd',
        json_path=eval_path
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True, collate_fn=xrd_collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=xrd_collate_fn)

    # 4. 初始化模型
    if model_class == 'XRDConvRegressionModel':
        model = XRDConvRegressionModel(
            embedding_dim=embedding_dim,
        )
    elif model_class == 'XRDFormulaModel':
        model = XRDFormulaModel(
            embedding_dim=embedding_dim,
        )
    elif model_class == 'XRDRegressionModel':
        model = XRDRegressionModel(
            embedding_dim=embedding_dim,
        )
    elif model_class == 'XRDClassificationModel':
        model = XRDClassificationModel(
        embedding_dim=embedding_dim,
    )
    elif model_class == 'XRDFormulaClassificationModel':
        model = XRDFormulaClassificationModel(
            embedding_dim=embedding_dim,
        )
        
    model = model.to(dtype=torch.float32)
    
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:  
    #         print(f"名称: {name}, 形状: {param.shape}")

    
    # 5. 创建训练器并训练
    if trainer_class == 'FormationEnergyTrainer':
        trainer = FormationEnergyTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=eval_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    elif trainer_class == 'RegressionTrainer':
        trainer = RegressionTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=eval_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    elif trainer_class == 'CrystalSystemClassificationTrainer':
        trainer = CrystalSystemClassificationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=eval_loader,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            log_dir=log_dir,
        )
        
    if model_path:
        trainer.load_model(model_path)
    
    # 6. 训练模型
    trainer.train(num_epochs=epochs, save_path=output_path)
    
    # 7. 使用训练好的模型获取embedding
    # test_data = torch.randn(10, input_dim)
    # embeddings = model.encode(test_data)
    # print(f"生成的embedding形状: {embeddings.shape}")
    
if __name__ == '__main__':
    import argparse
    
    # 1. 定义命令行解析器对象
    parser = argparse.ArgumentParser()
    
    # 2. 添加命令行参数
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_path', type=str, default=None)
    parser.add_argument('--eval_path', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default="logs")
    parser.add_argument('--model_class', type=str, default='XRDFormulaModel')
    parser.add_argument('--trainer_class', type=str, default='FormationEnergyTrainer')
    

    
    # 3. 从命令行中结构化解析参数 
    args = parser.parse_args()
    print(args)

    
    # epochs = args.epochs
    # batch_size = args.batch_size
    # temperature = args.temperature
    # learning_rate = args.learning_rate
    # weight_decay = args.weight_decay
    # embedding_dim = args.embedding_dim
    # projection_dim = args.projection_dim
    # use_qformer = args.use_qformer
    # output_path = args.output_path
    torch.multiprocessing.set_start_method('spawn')
    train(**vars(args))
    
    