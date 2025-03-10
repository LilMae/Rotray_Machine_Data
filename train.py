
from models.vae_model import VAE_model
from models.vqvae_model import VQVAE_model
from models.adhoc_layers import Classifier
from models.trainer import Trainer

from data.dataset import VibrationDataset, VibrationPipeline

from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader

import lightning as L
import torch
from pytorch_lightning.profilers import SimpleProfiler

from tqdm import tqdm
import argparse
import os

class CachedDataset(Dataset):
    def __init__(self, dataset, cache_path="cached_dataset.pt", force_reload=False):
        self.cache_path = cache_path
        self.data = []

        # ✅ 캐시된 파일이 존재하면 불러오기
        if not force_reload and os.path.exists(cache_path):
            print(f"📂 캐시된 데이터셋 불러오는 중: {cache_path}")
            self.data = torch.load(cache_path, weights_only=False)
            print("✅ 캐시된 데이터셋 로드 완료!")
        else:
            print("🔄 캐싱 중... (Dataset을 메모리에 로드하는 중)")
            self.data = []

            # ✅ 데이터셋을 (tensor, dict) 형태로 저장
            for i in tqdm(range(len(dataset)), desc="Dataset Caching", unit="samples"):
                tensor, meta_data = dataset[i]
                self.data.append((tensor, meta_data))  # 리스트에 추가

            print("✅ 캐싱 완료! 저장 중...")
            torch.save(self.data, cache_path)  # ✅ `.pt` 파일로 저장
            print(f"✅ 데이터셋 저장 완료: {cache_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]  # (tensor, meta_data) 형태 반환

def get_args():
    parser = argparse.ArgumentParser(description="Train Config")
    parser.add_argument('--project_name', type=str, help='snensor name for trainining', default='0219')
    # 데이터셋 관련 옵션들
    parser.add_argument('--target_channels', type=list, help='snensor name for trainining', default=['motor_x', 'motor_y'])
    parser.add_argument('--dataset_root', type=str, help='dataset_root', default='/home/lilmae/Desktop/Rotray_Machine_Data/data/new_dataset')
    parser.add_argument('--batch_size', type=int, help='batch_size', default=64)
    parser.add_argument('--target_dataset', type=list, help='dataset for training', default=['mfd', 'vat', 'vbl'])
    parser.add_argument('--target_class', type=list, help='classes for training', default=['looseness', 'normal', 'unbalance', 'misalignment', 'misalignment-horizontal', \
                                                                                            'misalignment-vertical','overhang', 'underhang', 'bearing-bpfi', 'bearing-bpfo'])
    # 모델 관련 옵션
    parser.add_argument('--num_hiddens', type=int, help='Encoder/Decoder final shape : num_hiddens x embedding_dim 최소 32', default=32)
    parser.add_argument('--embedding_dim', type=int, help='Encoder/Decoder final shape : num_hiddens x embedding_dim', default=32)
    parser.add_argument('--num_embeddings', type=int, help='after/before Encoder/Decoder shape : nu,_embeddings x embedding_dim', default=32)
    parser.add_argument('--num_residual_layers', type=int, help='number of residual layers for each residual blocks', default=3)
    parser.add_argument('--num_residual_hiddens', type=int, help='feature expension in residual connection : 2 means 2 times', default=2)
    parser.add_argument('--commitment_cost', type=float, help='in vqvae commitment_cost', default=0.25)
    parser.add_argument('--model_type', type=str, help='model type for training (vae, vqvae)', default='vqvae')
    parser.add_argument('--use_mapping_net', type=int, help='use mapping net for distangle', choices=[0, 1], default=1)
    parser.add_argument('--aux_classify_train', type=int, help='use classifier for training', choices=[0, 1], default=1)
    
    parser.add_argument('--use_log_scale', type=int, help='use classifier for training', choices=[0, 1], default=1)
    parser.add_argument('--fast_dev_run', type=bool, help='pytorch lightning fast_dev_run', default=False)
    
    return parser.parse_args()
    
def get_model(args, in_channels, in_length):

    if args.model_type =='vae':
        model = VAE_model(
            num_hiddens=args.num_hiddens,
            in_length = in_length,
            in_channels=in_channels,
            num_residual_layers=args.num_residual_layers,
            num_residual_hiddens=args.num_residual_hiddens,
            num_embeddings=args.num_embeddings,
            embedding_dim=args.embedding_dim,
            z_dim = args.num_embeddings,
            commitment_cost=args.commitment_cost,
            use_mapping_net=args.use_mapping_net
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
            commitment_cost=args.commitment_cost,
            use_mapping_net=args.use_mapping_net
        )
    else:
        print(f'Error unexpected model_type : {args.model_type}')
    
    return model
    
    
    
    
if __name__ == '__main__':
    args = get_args()
    args.use_mapping_net = bool(args.use_mapping_net)  # 1 → True, 0 → False
    args.aux_classify_train = bool(args.aux_classify_train)
    args.use_log_scale = bool(args.use_log_scale)
    
    pipeline = VibrationPipeline(
        harmonics=8,  
        points_per_harmonic=32,  
        smoothing_steps=1,  
        smoothing_param=0.1,
        log_scale=args.use_log_scale
    )
    
    print('Dataset Caching ...')
    num_workers = min(4, os.cpu_count() // 2)
    train_dataset = VibrationDataset(args.dataset_root, 
                                        target_dataset=args.target_dataset, 
                                        target_ch=args.target_channels, 
                                        target_class=args.target_class,
                                        transform=pipeline,
                                        )
    train_dataset_cache = CachedDataset(train_dataset, cache_path=f'./data/trainset_logsclae{args.use_log_scale}.pt')
    train_loader = DataLoader(train_dataset_cache, 
                                batch_size=args.batch_size, 
                                shuffle=True, 
                                num_workers=num_workers, 
                                pin_memory=True, 
                                prefetch_factor=4,
                                persistent_workers=True)
    
    test_dataset = VibrationDataset(args.dataset_root, 
                                    target_dataset=['dxai'], 
                                    target_ch=args.target_channels, 
                                    target_class=args.target_class, 
                                    transform=pipeline)
    test_dataset_cache = CachedDataset(test_dataset, cache_path=f'./data/testset_logsclae{args.use_log_scale}.pt')
    test_loader = DataLoader(test_dataset_cache, 
                                batch_size=args.batch_size, 
                                num_workers=num_workers, 
                                pin_memory=True, 
                                prefetch_factor=4,
                                persistent_workers=True)
    
    # Data Shape : B x 2 x 256
    sample_signal, sample_meta_data = train_dataset[0]
    in_channels = sample_signal.size(-2)
    in_length = sample_signal.size(-1)
    print(f'data shape : {in_channels} x {in_length}')
    model = get_model(args, in_channels, in_length)
    sample_input = torch.randn([4, in_channels, in_length])

    x_recon, KL = model.tesnor_transform_checker(sample_input)
    
    if args.aux_classify_train:
        classifier = Classifier(
            in_channels=args.num_embeddings,
            in_dim=args.embedding_dim,
            n_classes=len(args.target_class))
    else:
        classifier=None
    
    
    
    pl_model = Trainer(model=model,
                        model_type=args.model_type,
                        classifier=classifier,
                        classes = args.target_class,
                        batch_size = args.batch_size)
    
    # WandB Logger 설정
    
    wandb_logger = WandbLogger(
        project=args.project_name,  # 프로젝트 이름
        name=f'{args.model_type}_num_hidden{args.num_hiddens}_\
                embedding_dim{args.embedding_dim}_\
                num_embeddings{args.num_embeddings}_\
                num_residual_layers{args.num_residual_layers}_\
                num_residual_hiddens{args.num_residual_hiddens}_\
                use_mapping_net{args.use_mapping_net}_\
                aux_classify_train{args.aux_classify_train}_\
                use_log_scale{args.use_log_scale}',
        save_dir='results',
        log_model=True,  # 모델 구조 로깅
        config = vars(args)
    )
    
    profiler = SimpleProfiler()
    trainer = L.Trainer(
        max_epochs=200,
        logger=wandb_logger,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
        profiler=profiler,
        precision=32
    )
    
    trainer.fit(pl_model, train_loader, test_loader)
    
    # 실행 후 데이터 로딩 시간이 오래 걸리는지 확인
    print(trainer.profiler.summary())
