import os
import random
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics as sm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset import ScannetDatasetWholeScene
from model.mlp_former import MLP_Former
from model.simpleview import SimpleView
from model.attention_fusion import CanAttention,CrossAttentionFusion
from pytorch_metric_learning import losses

from addict import Dict
import logging
import argparse
import sys
from datetime import datetime
import json

# ========================== 配置日志系统 ==========================
def setup_logging(log_dir="logs", ratio=None):
    """设置日志系统，同时输出到控制台和文件"""
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 如果指定了比例，在日志文件名中包含比例信息
    if ratio is not None:
        log_file = os.path.join(log_dir, f"training_{int(ratio*100)}percent_{timestamp}.log")
    else:
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # 创建日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file

# ========================== 配置设备 ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========================== 随机种子设置函数 ==========================
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ========================== 参数解析 ==========================
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Point Render Species Classification')
    
    # 训练参数
    parser.add_argument('--epoch', type=int, default=1600)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=5132)
    
    # 网络参数
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--numviews', type=int, default=6)
    parser.add_argument('--num_class', type=int, default=5)
    parser.add_argument('--num_feature', type=int, default=512)
    
    # 优化器参数
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--supcon_temperature', type=float, default=0.08)
    parser.add_argument('--label_smoothing', type=float, default=0.2)
    
    # 训练策略
    parser.add_argument('--supcon_epochs', type=int, default=800)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    
    # 日志参数
    parser.add_argument('--log_dir', type=str, default="logs")
    
    # 训练数据比例
    parser.add_argument('--train_ratios', nargs='+', type=float, default=[0.7, 0.8, 0.9],
                       help='List of training data ratios to use')
    # 断点训练参数
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    return parser.parse_known_args()[0]

# ========================== 模型和优化器设置 ==========================
def setup_models_and_optimizers(args, device):
    """初始化模型和优化器"""
    # 初始化模型
    point_feature = MLP_Former().to(device)
    image_feature = SimpleView().to(device)
    feature_fuse = CanAttention().to(device)
    
    # 初始化优化器
    point_optimizer = torch.optim.AdamW(
        point_feature.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    image_optimizer = torch.optim.AdamW(
        image_feature.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    fuse_optimizer = torch.optim.AdamW(
        feature_fuse.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    point_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        point_optimizer, 
        T_max=args.epoch, 
        eta_min=args.min_lr
    )
    image_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        image_optimizer, 
        T_max=args.epoch, 
        eta_min=args.min_lr
    )
    fuse_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        fuse_optimizer, 
        T_max=args.epoch, 
        eta_min=args.min_lr
    )
    
    # 损失函数
    cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    supcon_loss = losses.SupConLoss(temperature=args.supcon_temperature)
    
    return {
        'models': {
            'point': point_feature,
            'image': image_feature,
            'fuse': feature_fuse
        },
        'optimizers': {
            'point': point_optimizer,
            'image': image_optimizer,
            'fuse': fuse_optimizer
        },
        'schedulers': {
            'point': point_scheduler,
            'image': image_scheduler,
            'fuse': fuse_scheduler
        },
        'loss_functions': {
            'cross_entropy': cross_entropy_loss,
            'supcon': supcon_loss
        }
    }
def save_checkpoint(setup_dict, epoch, args, ratio, best_val_acc, best_test_acc, 
                   best_val_epoch, best_test_epoch, checkpoint_dir, filename):
    """保存训练检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'models': {
            'point': setup_dict['models']['point'].state_dict(),
            'image': setup_dict['models']['image'].state_dict(),
            'fuse': setup_dict['models']['fuse'].state_dict(),
        },
        'optimizers': {
            'point': setup_dict['optimizers']['point'].state_dict(),
            'image': setup_dict['optimizers']['image'].state_dict(),
            'fuse': setup_dict['optimizers']['fuse'].state_dict(),
        },
        'schedulers': {
            'point': setup_dict['schedulers']['point'].state_dict(),
            'image': setup_dict['schedulers']['image'].state_dict(),
            'fuse': setup_dict['schedulers']['fuse'].state_dict(),
        },
        'best_val_acc': best_val_acc,
        'best_test_acc': best_test_acc,
        'best_val_epoch': best_val_epoch,
        'best_test_epoch': best_test_epoch,
        'args': vars(args),
        'ratio': ratio,
        'random_seed': args.random_seed,
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path
def load_checkpoint(checkpoint_path, setup_dict, device):
    """加载训练检查点"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    setup_dict['models']['point'].load_state_dict(checkpoint['models']['point'])
    setup_dict['models']['image'].load_state_dict(checkpoint['models']['image'])
    setup_dict['models']['fuse'].load_state_dict(checkpoint['models']['fuse'])
    
    # 加载优化器状态
    setup_dict['optimizers']['point'].load_state_dict(checkpoint['optimizers']['point'])
    setup_dict['optimizers']['image'].load_state_dict(checkpoint['optimizers']['image'])
    setup_dict['optimizers']['fuse'].load_state_dict(checkpoint['optimizers']['fuse'])
    
    # 加载调度器状态
    setup_dict['schedulers']['point'].load_state_dict(checkpoint['schedulers']['point'])
    setup_dict['schedulers']['image'].load_state_dict(checkpoint['schedulers']['image'])
    setup_dict['schedulers']['fuse'].load_state_dict(checkpoint['schedulers']['fuse'])
    
    return checkpoint
# ========================== 训练和评估函数 ==========================
def train_epoch(models, optimizers, loss_fns, data_loader, args, device, epoch):
    """训练一个epoch"""
    models['point'].train()
    models['image'].train()
    models['fuse'].train()
    
    running_loss = 0
    all_preds = []
    all_labels = []
    
    # 使用tqdm显示进度条
    pbar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{args.epoch} [Train]')
    
    for batch_idx, batch_data in enumerate(pbar):
        # 清零梯度
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        # 解析批量数据
        # 根据ScannetDatasetWholeScene的实现，可能需要调整这里的解析方式
        # 假设返回格式为 (trad_features, images, points, labels)
        trad_features, images, points, labels = batch_data
        points = points.squeeze(1).float().to(device)
        images = images.squeeze(1).float().to(device)
        trad_features = trad_features.squeeze(1).to(device).float()
        # print("points shape: ", points.shape)
        # print("images shape: ", images.shape)
        # print("trad_features shape: ", trad_features.shape)
        # 转移到设备
        images = images.to(device)
        points = points.to(device)
        labels = labels.to(device)
        
        # 前向传播
        image_features = models['image'](images)
        point_features = models['point'](points)
        _, features, pred = models['fuse'](image_features, point_features, trad_features)
        
        # 计算损失
        if epoch < args.supcon_epochs:
            # 使用SupCon损失
            loss = loss_fns['supcon'](features, labels)
            cross_loss = torch.tensor(0.0)
            supcon_loss = loss
        else:
            # 使用交叉熵损失
            cross_loss = loss_fns['cross_entropy'](pred, labels)
            supcon_loss = torch.tensor(0.0)
            loss = cross_loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(models['point'].parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(models['image'].parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(models['fuse'].parameters(), max_norm=1.0)
        
        # 更新参数
        for optimizer in optimizers.values():
            optimizer.step()
        
        # 记录损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(pred, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.5f}',
            'Cross': f'{cross_loss.item():.5f}',
            'SupCon': f'{supcon_loss.item():.5f}'
        })
    
    # 计算epoch准确率
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_preds == all_labels)
    avg_loss = running_loss / len(data_loader)
    
    return accuracy, avg_loss

def evaluate(models, data_loader, loss_fns, args, device, epoch, phase='val' ):
    """评估模型"""
    models['point'].eval()
    models['image'].eval()
    models['fuse'].eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'[{phase.capitalize()}] Evaluation')
        
        for batch_idx, batch_data in enumerate(pbar):
            # 解析批量数据
            trad_features, images, points, labels = batch_data
            points = points.squeeze(1).float().to(device)
            images = images.squeeze(1).float().to(device)
            trad_features = trad_features.squeeze(1).to(device).float()

            # 转移到设备
            # trad_features = trad_features.to(device).float()
            images = images.to(device)
            points = points.to(device)
            labels = labels.to(device)
            
            # 前向传播
            image_features = models['image'](images)
            point_features = models['point'](points)
            _, features, pred = models['fuse'](image_features, point_features, trad_features)
            
            # 计算损失
            # if phase == 'val' or phase == 'test':
            #     loss = loss_fn['cross_entropy'](pred, labels)
            #     total_loss += loss.item()
            if epoch < args.supcon_epochs:
            # 使用SupCon损失
                loss = loss_fns['supcon'](features, labels)
                cross_loss = torch.tensor(0.0)
                # supcon_loss = loss
            else:
                # 使用交叉熵损失
                cross_loss = loss_fns['cross_entropy'](pred, labels)
                supcon_loss = torch.tensor(0.0)
                loss = cross_loss
            # 记录预测结果
            total_loss += loss.item()
            _, predicted = torch.max(pred, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.5f}'})
    
    # 计算评估指标
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = np.mean(all_preds == all_labels)
    confusion_mat = sm.confusion_matrix(all_labels, all_preds, normalize='true')
    class_acc = np.diag(confusion_mat)
    avg_acc = np.mean(class_acc)
    avg_loss = total_loss / len(data_loader)
    
    return accuracy, class_acc, avg_acc, avg_loss

# ========================== 创建子数据集 ==========================
def create_subset(dataset, ratio, seed=42):
    """创建指定比例的子数据集"""
    # 设置随机种子以确保可重复性
    set_random_seed(seed)
    
    # 计算子集大小
    dataset_size = len(dataset)
    subset_size = int(dataset_size * ratio)
    
    # 随机选择索引
    indices = list(range(dataset_size))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    
    # 创建子集
    subset = Subset(dataset, subset_indices)
    
    return subset, subset_size

# ========================== 主函数 ==========================
def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_random_seed(args.random_seed)
    
    # ========================== 数据路径 ==========================
    train_root = "data/train/"
    val_root = "data/val/"
    test_root = "data/test/"
    metadir = "data/meta/tree-meta.csv"
    
    # ---------------- 初始化完整数据集 ----------------
    full_train_dataset = ScannetDatasetWholeScene(
        root=train_root,
        metadata_file=metadir,
        aug=True,
        transforms=['rotation','translation','scaling'],
        seed=args.random_seed
    )
    
    val_dataset = ScannetDatasetWholeScene(
        root=val_root,
        metadata_file=metadir,
        aug=False,
        transforms=[],
        seed=args.random_seed
    )
    
    test_dataset = ScannetDatasetWholeScene(
        root=test_root,
        metadata_file=metadir,
        aug=False,
        transforms=[],
        seed=args.random_seed
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)
    
    # 存储所有比例的结果
    all_results = {}

    # 检查是否从检查点恢复
    start_ratio_idx = 0
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
    # 从检查点恢复训练
        try:
            # 先加载检查点获取比例信息
            temp_checkpoint = torch.load(args.resume, map_location=device)
            resume_ratio = temp_checkpoint['ratio']
            
            # 找到对应的比例索引
            for i, ratio in enumerate(args.train_ratios):
                if ratio == resume_ratio:
                    start_ratio_idx = i
                    break
            
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            logger.info(f"Resuming from ratio: {resume_ratio}, epoch: {temp_checkpoint['epoch']+1}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return
    # 循环不同的训练数据比例
    for ratio in args.train_ratios:
        # 设置当前比例的日志
        logger, log_file = setup_logging(args.log_dir, ratio)
        logger.info(f"Training with {int(ratio*100)}% of training data")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Command line arguments: {args}")
        
        # 创建当前比例的子数据集
        logger.info(f"Creating subset with {int(ratio*100)}% of training data...")
        train_subset, subset_size = create_subset(full_train_dataset, ratio, args.random_seed)
        train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=False)
        
        logger.info(f"Full training dataset size: {len(full_train_dataset)}")
        logger.info(f"Subset training dataset size: {subset_size}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # 初始化模型、优化器和损失函数
        logger.info("Initializing models, optimizers and loss functions...")
        setup_dict = setup_models_and_optimizers(args, device)
        
        # 记录模型结构
        logger.info("Model architectures:")
        for name, model in setup_dict['models'].items():
            logger.info(f"{name} model: {model}")
        
        # 初始化TensorBoard
        from torch.utils.tensorboard import SummaryWriter
        tb_log_dir = os.path.join(args.log_dir, "tensorboard", f"{int(ratio*100)}percent")
        writer = SummaryWriter(log_dir=tb_log_dir)
        logger.info(f"TensorBoard logs will be saved to: {tb_log_dir}")

        # 创建检查点目录
        checkpoint_dir = os.path.join(args.log_dir, 'checkpoints', f"{int(ratio*100)}percent")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练循环
        best_val_acc = 0
        best_test_acc = 0
        best_val_epoch = 0
        best_test_epoch = 0

        # # 如果从检查点恢复并且是当前比例
        # if args.resume and ratio_idx == start_ratio_idx:
        #     try:
        #         # 加载检查点
        #         checkpoint = load_checkpoint(args.resume, setup_dict, device)
        #         start_epoch_for_ratio = checkpoint['epoch'] + 1
        #         best_val_acc = checkpoint['best_val_acc']
        #         best_test_acc = checkpoint['best_test_acc']
        #         best_val_epoch = checkpoint['best_val_epoch']
        #         best_test_epoch = checkpoint['best_test_epoch']
                
        #         logger.info(f"Resumed from epoch {start_epoch_for_ratio}")
        #         logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}")
        #         logger.info(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_test_epoch}")
                
        #     except Exception as e:
        #         logger.error(f"Error loading checkpoint: {e}")
        #         return
        logger.info("Starting training...")
        
        for epoch in range(args.epoch):
            logger.info('=' * 100)
            logger.info(f"Epoch {epoch+1}/{args.epoch} - {int(ratio*100)}% data")
            
            # 训练
            train_acc, train_loss = train_epoch(
                setup_dict['models'],
                setup_dict['optimizers'],
                setup_dict['loss_functions'],
                train_loader,
                args,
                device,
                epoch
            )
            
            # 更新学习率
            for name, scheduler in setup_dict['schedulers'].items():
                old_lr = scheduler.get_last_lr()[0]
                scheduler.step()
                new_lr = scheduler.get_last_lr()[0]
                if old_lr != new_lr:
                    logger.info(f"{name} learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
            
            # 验证
            val_acc, val_class_acc, val_avg_acc, val_loss = evaluate(
                setup_dict['models'],
                val_loader,
                setup_dict['loss_functions'],
                args,
                device,
                epoch,
                'val'
            )
            
            # 测试
            test_acc, test_class_acc, test_avg_acc, test_loss = evaluate(
                setup_dict['models'],
                test_loader,
                setup_dict['loss_functions'],
                args,
                device,
                epoch,
                'test'
            )
            
            # 记录到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            
            # 保存最佳模型
            if val_acc > best_val_acc and epoch > args.supcon_epochs:
                best_val_acc = val_acc
                best_val_epoch = epoch
                val_model_dir = os.path.join(args.log_dir, 'weights', f"{int(ratio*100)}percent", 'val')
                os.makedirs(val_model_dir, exist_ok=True)
                val_model_path = os.path.join(val_model_dir, 'best_val_model.pth')
                
                torch.save({
                    'point_feature': setup_dict['models']['point'].state_dict(),
                    'image_feature': setup_dict['models']['image'].state_dict(),
                    'feature_fuse': setup_dict['models']['fuse'].state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc
                }, val_model_path)
                logger.info(f'Saved new best VAL model at epoch {epoch} with accuracy {val_acc:.4f} to {val_model_path}')
            
            if test_acc > best_test_acc and epoch > args.supcon_epochs:
                best_test_acc = test_acc
                best_test_epoch = epoch
                test_model_dir = os.path.join(args.log_dir, 'weights', f"{int(ratio*100)}percent", 'test')
                os.makedirs(test_model_dir, exist_ok=True)
                test_model_path = os.path.join(test_model_dir, 'best_test_model.pth')
                
                torch.save({
                    'point_feature': setup_dict['models']['point'].state_dict(),
                    'image_feature': setup_dict['models']['image'].state_dict(),
                    'feature_fuse': setup_dict['models']['fuse'].state_dict(),
                    'epoch': epoch,
                    'test_acc': test_acc
                }, test_model_path)
                logger.info(f'Saved new best TEST model at epoch {epoch} with accuracy {test_acc:.4f} to {test_model_path}')
                # 定期保存检查点
            if (epoch) % args.checkpoint_interval == 0:
                checkpoint_path = save_checkpoint(
                    setup_dict, epoch, args, ratio, 
                    best_val_acc, best_test_acc, 
                    best_val_epoch, best_test_epoch,
                    checkpoint_dir, f'checkpoint_newest_percentage_{ratio*100}.pth'
                )
            logger.info(f'Saved checkpoint to: {checkpoint_path}')
            # 每个epoch记录一次详细结果
            logger.info(f'Epoch: {epoch}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}')
            logger.info(f'Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}')
            logger.info(f'Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}')
            logger.info(f'Val Class Acc: {val_class_acc}, Val Avg Acc: {val_avg_acc:.4f}')
            logger.info(f'Test Class Acc: {test_class_acc}, Test Avg Acc: {test_avg_acc:.4f}')
            logger.info('=' * 100)
        
        # 训练结束
        logger.info(f"Training with {int(ratio*100)}% data completed.")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_val_epoch}")
        logger.info(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_test_epoch}")
        
        # 存储当前比例的结果
        all_results[f"{int(ratio*100)}%"] = {
            "best_val_acc": float(best_val_acc),
            "best_val_epoch": int(best_val_epoch),
            "best_test_acc": float(best_test_acc),
            "best_test_epoch": int(best_test_epoch),
            "training_samples": int(subset_size)
        }
        
        # 关闭TensorBoard写入器
        writer.close()
        
        # 清除当前日志处理器，为下一个比例做准备
        logger.handlers.clear()
    
    # 保存所有比例的结果到JSON文件
    results_file = os.path.join(args.log_dir, "all_results.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # 打印所有比例的结果摘要
    print("\n" + "="*80)
    print("SUMMARY OF ALL TRAINING RESULTS")
    print("="*80)
    for ratio, results in all_results.items():
        print(f"{ratio} data: Val Acc = {results['best_val_acc']:.4f}, Test Acc = {results['best_test_acc']:.4f}")
    
    return all_results

if __name__ == '__main__':
    results = main()