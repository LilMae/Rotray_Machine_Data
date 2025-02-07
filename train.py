from models.ae_model import AE_model
from models.vae_model import VAE_model
from models.vqvae_model import VQVAE_model

from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from dataset import VibrationDataset, VibrationPipeline

from models.trainer import Trainer

import argparse
import os
import lightning as L
import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Train Config")
    
    # 데이터셋 관련 옵션들
    parser.add_argument('--target_channels', type=list, help='snensor name for trainining', default=['motor_x', 'motor_y'])
    parser.add_argument('--dataset_root', type=str, help='dataset_root', default=os.path.join(os.getcwd(), 'data', 'new_dataset'))
    parser.add_argument('--target_dataset', type=list, help='dataset for training', default=['mfd', 'vat', 'vbl'])
    parser.add_argument('--target_class', type=list, help='classes for training', default=['looseness', 'normal', 'unbalance', 'misalignment', 'misalignment-horizontal', \
                                                                                            'misalignment-vertical','overhang', 'underhang', 'bearing-bpfi', 'bearing-bpfo'])
    
    # 모델 관련 옵션들
    parser.add_argument('--num_hiddens', type=int, help='Encoder/Decoder final shape : num_hiddens x embedding_dim', default=32)
    parser.add_argument('--embedding_dim', type=int, help='Encoder/Decoder final shape : num_hiddens x embedding_dim', default=32)
    parser.add_argument('--num_embeddings', type=int, help='after/before Encoder/Decoder shape : nu,_embeddings x embedding_dim', default=32)
    parser.add_argument('--num_residual_layers', type=int, help='number of residual layers for each residual blocks', default=3)
    parser.add_argument('--num_residual_hiddens', type=int, help='feature expension in residual connection : 2 means 2 times', default=2)
    parser.add_argument('--commitment_cost', type=float, help='in vqvae commitment_cost', default=0.25)
    
    parser.add_argument('--training_mode', type=float, help='type of training (not implement yet)', default='recon')
    parser.add_argument('--model_type', type=float, help='model type for training (ae, vae, vqvae)', default=0.25)
    parser.add_argument('--fast_dev_run', type=bool, help='pytorch lightning fast_dev_run', default=False)
    
    
def get_model(args, in_channels, in_length):
    if args.model_type == 'ae':
        model = AE_model(
            num_hiddens=args.num_hiddens,
            in_length = in_length,
            in_channels=in_channels,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.num_residual_hiddens,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost
        )
    elif args.model_type =='vae':
        model = VAE_model(
            num_hiddens=args.num_hiddens,
            in_length = in_length,
            in_channels=in_channels,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.num_residual_hiddens,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost
        )
    elif args.model_type =='vqvae':
        model = VQVAE_model(
            num_hiddens=args.num_hiddens,
            in_length = in_length,
            in_channels=in_channels,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.num_residual_hiddens,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            commitment_cost=args.commitment_cost
        )
    else:
        print(f'Error unexpected model_type : {args.model_type}')
    
    return model
    
    
    
    
if __name__ == '__main__':
    args = get_args()
    
    pipeline = VibrationPipeline(
        harmonics=8,  
        points_per_harmonic=32,  
        smoothing_steps=1,  
        smoothing_param=0.1  
    )
    train_dataset = VibrationDataset(args.dataset_root, target_dataset=args.target_dataset, target_ch=args.target_channels, target_class=args.target_class, transform=pipeline)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=19)
    test_dataset = VibrationDataset(args.dataset_root, target_dataset=['dxai'], target_ch=args.target_channels, target_class=args.target_class, transform=pipeline)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=19)
    # Data Shape : B x 2 x 256
    sample_signal, sample_meta_data = train_dataset[0]
    in_channels = sample_signal.size(-2)
    in_length = sample_signal.size(-1)
    
    model = get_model(args, in_channels, in_length)
    pl_model = Trainer(model=model, 
                        training_mode=args.training_mode,
                        model_type=args.model_type)

    # WandB Logger 설정
    wandb_logger = WandbLogger(
        project="Vibration-Reconstruction",  # 프로젝트 이름
        name=f'{args.model_type}',  # 실험 이름
        save_dir='./results',
        log_model=True,  # 모델 구조 로깅
        config = vars(args)
    )
    
    trainer = L.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run
    )
    
    trainer.fit(pl_model, train_loader, test_loader)