import wandb
import os
import lightning as L
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from models import AutoEncoder1D
from utils import LinearTransformationModule
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from dataset import VibrationDataset, VibrationPipeline

class LightningAE(L.LightningModule):
    def __init__(self, ae_model, ltm, training_mode='recon'):
        super(LightningAE, self).__init__()
        
        self.model = ae_model
        self.ltm = ltm
        self.training_mode = training_mode
    
        self.first_val_step = True  # 첫 번째 validation step을 체크하는 플래그
    
    def configure_optimizers(self):
        if self.training_mode == 'recon':
            optimzier = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            return optimzier
        elif self.training_mode =='transfer':
            optimizer = torch.optim.Adam(self.ltm.parameters(), lr=1e-3)
            return optimizer
    
    def training_step(self, batch, batch_idx):
        
        if self.training_mode == 'recon':
            signal_data, meta_data = batch
            recon_loss = ae_model.recon_loss(signal_data, with_recon=False)

            self.log("train/recon_loss", recon_loss, prog_bar=True, logger=True)

            
            return recon_loss
        
        elif self.training_mode =='transfer':
            content_input, style_input = batch
            
            content_feature = ae_model.encode(content_input, before_vector=True)
            style_feature = ae_model.encode(style_input, before_vector=True)

            tranfer_feature, transmatrix, KL = ltm(content_feature=content_feature, style_feature=style_feature, transfer=True)
            transfer_out = ae_model.decode(tranfer_feature)

            style_loss, content_loss = ae_model.style_content_loss(style_input=style_input, content_input=content_input, transfer_out=transfer_out)

            total_loss = style_loss + content_loss + KL.sum()
            
            self.log("train/total_loss", total_loss, prog_bar=True, logger=True)
            self.log("train/style_loss", style_loss, logger=True)
            self.log("train/content_loss", content_loss, logger=True)
            self.log("train/KL_loss", KL.sum(), logger=True)
            
            return total_loss
        
    def validation_step(self, batch, batch_idx):
        if self.training_mode == 'recon':
            signal_data, meta_data = batch
            recon_loss, recon_output = self.model.recon_loss(signal_data, with_recon=True)
            self.log("val/recon_loss", recon_loss, prog_bar=True, logger=True)
            
            # 첫 번째 validation step에서 시각화 수행
            if self.first_val_step:
                self.first_val_step = False  # 이후 step에서는 실행하지 않도록 설정
                self.plot_reconstruction(signal_data, recon_output)

            return recon_loss
        
        elif self.training_mode == 'transfer':
            content_input, style_input = batch
            
            content_feature = self.model.encode(content_input, before_vector=True)
            style_feature = self.model.encode(style_input, before_vector=True)

            transfer_feature, transmatrix, KL = self.ltm(
                content_feature=content_feature, 
                style_feature=style_feature, 
                transfer=True
            )
            transfer_out = self.model.decode(transfer_feature)

            style_loss, content_loss = self.model.style_content_loss(
                style_input=style_input, 
                content_input=content_input, 
                transfer_out=transfer_out
            )

            total_loss = style_loss + content_loss + KL.sum()

            # 🟢 WandB에 다양한 Validation Loss 기록
            self.log("val/total_loss", total_loss, prog_bar=True, logger=True)
            self.log("val/style_loss", style_loss, logger=True)
            self.log("val/content_loss", content_loss, logger=True)
            self.log("val/KL_loss", KL.sum(), logger=True)

            # 첫 번째 validation step에서 시각화 수행
            if self.first_val_step:
                self.first_val_step = False  # 이후 step에서는 실행하지 않도록 설정
                self.plot_signals(content_input, style_input, transfer_out)

            return total_loss

    def plot_reconstruction(self, original_input, recon_output):
        """Validation에서 Reconstruction 결과를 WandB에 시각화"""
        num_samples = 3  # 3개의 샘플을 시각화
        fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6))

        for i in range(num_samples):
            # 2채널 신호 가져오기
            original_signal = original_input[i].detach().cpu().numpy()
            recon_signal = recon_output[i].detach().cpu().numpy()
            
            # x축 (시계열 데이터 기준)
            time = np.arange(original_signal.shape[-1])

            # 2채널을 각각 플롯
            for ch in range(2):
                axes[i, 0].plot(time, original_signal[ch], label=f"Ch {ch+1}")
                axes[i, 1].plot(time, recon_signal[ch], label=f"Ch {ch+1}")

            axes[i, 0].set_title(f"Original Signal {i+1}")
            axes[i, 1].set_title(f"Reconstructed Signal {i+1}")

            for j in range(2):
                axes[i, j].legend()
                axes[i, j].grid()

        plt.tight_layout()

        # WandB에 플롯 업로드
        wandb.log({"Reconstruction Comparison": wandb.Image(fig)})
        plt.close(fig)

    def plot_signals(self, content_input, style_input, transfer_out):
        """Validation의 첫 번째 step에서 신호 데이터를 플롯하여 WandB에 로깅"""
        num_samples = 3  # 3개의 샘플을 시각화
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 8))

        for i in range(num_samples):
            # 2채널 신호 가져오기 (배치에서 i번째 샘플)
            content_signal = content_input[i].detach().cpu().numpy()
            style_signal = style_input[i].detach().cpu().numpy()
            transfer_signal = transfer_out[i].detach().cpu().numpy()
            
            # x축 (시계열 데이터 기준)
            time = np.arange(content_signal.shape[-1])  

            # 2채널을 각각 그리기
            for ch in range(2):  # 2개의 채널 (Stereo Signal)
                axes[i, 0].plot(time, content_signal[ch], label=f"Ch {ch+1}")
                axes[i, 1].plot(time, style_signal[ch], label=f"Ch {ch+1}")
                axes[i, 2].plot(time, transfer_signal[ch], label=f"Ch {ch+1}")

            axes[i, 0].set_title(f"Content Signal {i+1}")
            axes[i, 1].set_title(f"Style Signal {i+1}")
            axes[i, 2].set_title(f"Transfer Output {i+1}")

            for j in range(3):
                axes[i, j].legend()
                axes[i, j].grid()

        plt.tight_layout()

        # WandB에 플롯 업로드
        wandb.log({"Signal Reconstruction": wandb.Image(fig)})
        plt.close(fig)

if __name__ == '__main__':
    # 1. AutoEncoder 모델과 Style Transfer를 수행할  Linear Transfromation Module
    ae_model = AutoEncoder1D()
    ltm = LinearTransformationModule()

    lightning_ae = LightningAE(ae_model=ae_model, ltm=ltm, training_mode='recon')
    
    # WandB Logger 설정
    wandb_logger = WandbLogger(
        project="Vibration-AutoEncoder",  # 프로젝트 이름
        name="AE_Reconstruction",  # 실험 이름
        log_model=True  # 모델 구조 로깅
    )
    
    # Example parameters
    target_channels = ['motor_x', 'motor_y']  # Example channel names
    dxai_root = os.path.join(os.getcwd(), 'data', 'new_dataset')  # Path to vibration data
    target_dataset=['iis', 'mfd', 'vat', 'vbl']
    target_class=['looseness', 'normal', 'unbalance','misalignment', 'horizontal-misalignment', 'vertical-misalignment',
                    'overhang', 'underhang', 'bpfi', 'bpfo', 'bearing']

    pipeline = VibrationPipeline(
        harmonics=8,  # Number of harmonics to extract
        points_per_harmonic=32,  # 30 points per harmonic
        smoothing_steps=1,  # Apply low-pass filter 3 times
        smoothing_param=0.1  # Lowpass cutoff frequency as 0.2 * Nyquist
    )
    train_dataset = VibrationDataset(dxai_root, target_dataset=target_dataset, target_ch=target_channels, target_class=target_class, transform=pipeline)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=19)
    test_dataset = VibrationDataset(dxai_root, target_dataset=['dxai'], target_ch=target_channels, target_class=target_class, transform=pipeline)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=19)
    
    # Trainer 설정
    trainer = L.Trainer(
        max_epochs=50,  # 최대 학습 Epochs
        logger=wandb_logger,  # WandB 로깅 추가
        log_every_n_steps=10  # 매 10 스텝마다 로그 기록
    )
    
    # 모델 학습
    trainer.fit(lightning_ae, train_loader, test_loader)