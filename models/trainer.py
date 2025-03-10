import wandb
import os
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

class ReconLoss(nn.Module):
    def __init__(self, loss_name='mae'):
        super().__init__()
        if loss_name == 'mae':
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()
    
    def forward(self, y_hat, y):
        return self.loss(y_hat, y)

class Trainer(L.LightningModule):
    def __init__(self, model, batch_size, model_type='vae', classifier=None, classes=None):
        super(Trainer, self).__init__()
        
        self.model = model
        self.model_type = model_type
        self.batch_size = batch_size
        self.mse_loss = ReconLoss()
    
        self.classifier = classifier
        self.classes = classes
        self.y_true, self.y_pred, self.encoded_features, self.labels = [], [], [], []
        if self.classifier:
            self.criterion_class = nn.CrossEntropyLoss()
        self.input_signals = None
        self.first_val_step = True  # 첫 번째 validation step을 체크하는 플래그
        
    def label_to_index(self, labels):
        """Convert labels to indices."""
        class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        return torch.tensor([class_to_idx[label] for label in labels], device=self.device)
    
    def configure_optimizers(self):
        optimzier = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimzier

    def compute_recon_loss(self, signal_data, meta_data):
        
        recon, distribution_loss = self.model(signal_data)
        recon_loss = self.mse_loss(recon, signal_data)
        
        beta = min(1.0, self.current_epoch / 100)  # 예: 50 epoch 동안 0 -> 1로 증가
        lambda_kl = 0.1  # KL loss를 10%로 반영
        total_loss = recon_loss + lambda_kl * beta * distribution_loss
        distribution_loss = lambda_kl*beta*distribution_loss
        total_loss = recon_loss.sum() + distribution_loss.sum()
        return total_loss, recon_loss, distribution_loss
    
    def compute_classification_loss(self, signal_data, meta_data):
        z = self.model.encode(signal_data, True)
        
        pred = self.classifier(z)
        indices = self.label_to_index(meta_data['class_name'])
        return self.criterion_class(pred, indices), pred, indices
    
    def training_step(self, batch, batch_idx):
        signal_data, meta_data = batch
        total_loss, recon_loss, distribution_loss = self.compute_recon_loss(signal_data, meta_data)
        self.log("train/recon_loss", recon_loss, batch_size=self.batch_size)
        self.log("train/distribution_loss", distribution_loss, batch_size=self.batch_size)
        
        if self.classifier:
            class_loss, _, _ = self.compute_classification_loss(signal_data, meta_data)
            self.log("train/classification_loss", class_loss, batch_size=self.batch_size)
            total_loss += class_loss
        
        self.log("train/total_loss", total_loss, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch, batch_idx):
        self.input_signals = batch[0]  # Store input signals for visualization
        signal_data, meta_data = batch
        total_loss, recon_loss, distribution_loss = self.compute_recon_loss(signal_data, meta_data)
        self.log("val/recon_loss", recon_loss, batch_size=self.batch_size)
        self.log("val/distribution_loss", distribution_loss, batch_size=self.batch_size)

        z = self.model.encode(signal_data)
        z = z.view(z.shape[0], -1)  # 배치 차원 유지
        self.encoded_features.append(z.detach().cpu())  # GPU에서 CPU로 이동하여 저장
        self.labels.append(self.label_to_index(meta_data['class_name']).detach().cpu())

        if self.classifier:
            class_loss, pred, indices = self.compute_classification_loss(signal_data, meta_data)
            self.log("val/classification_loss", class_loss, batch_size=self.batch_size)
            total_loss += class_loss
            self.y_true.extend(indices.cpu().numpy())
            self.y_pred.extend(torch.argmax(pred, dim=1).cpu().numpy())
        
        self.log("val/total_loss", total_loss, batch_size=self.batch_size)
        return total_loss

    def on_validation_epoch_end(self):
        self.visualize_latent_space()
        self.visualize_reconstructions()
        if self.classifier and self.y_true:
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_true, self.y_pred, average='macro', zero_division=1)
            self.log("val/precision", precision, self.batch_size)
            self.log("val/recall", recall, self.batch_size)
            self.log("val/f1_score", f1, self.batch_size)
            self.log_confusion_matrix()
            self.y_true.clear()
            self.y_pred.clear()

    def log_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

    def visualize_latent_space(self):
        if not self.encoded_features:
            return
        encoded_features_all = torch.cat(self.encoded_features, dim=0).numpy()
        labels_all = torch.cat(self.labels, dim=0).numpy()
        label_names = np.array([self.classes[idx] for idx in labels_all])
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(encoded_features_all)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_features)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42, init='random')
        tsne_result = tsne.fit_transform(scaled_features)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=label_names, palette="Set1", ax=axes[0])
        axes[0].set_title("PCA Visualization")
        sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=label_names, palette="Set1", ax=axes[1])
        axes[1].set_title("t-SNE Visualization")
        plt.tight_layout()
        self.logger.experiment.log({"Feature Encoding Visualization": wandb.Image(fig)})
        plt.close(fig)
        self.encoded_features.clear()
        self.labels.clear()
        
        if not self.y_pred:
            return
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        self.logger.experiment.log({"val/confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        
    def visualize_reconstructions(self):
        if self.input_signals is None:
            return
        num_samples = 3
        fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))  # 2개의 플롯(채널별) 생성
        for i in range(num_samples):
            original_signal = self.input_signals[i].unsqueeze(0).to(self.device)
            recon_signal, _ = self.model(original_signal.to(self.device))
            
            original_signal = original_signal.squeeze(0).detach().cpu().numpy()
            recon_signal = recon_signal.squeeze(0).detach().cpu().numpy()
            time = np.arange(original_signal.shape[-1])
            
            for ch in range(2):  # 2개의 채널
                axes[ch, i].plot(time, original_signal[ch], label="Original", linestyle='-')
                axes[ch, i].plot(time, recon_signal[ch], label="Reconstructed", linestyle='dashed')
                axes[ch, i].set_title(f"Sample {i+1} - Channel {ch+1}")
                axes[ch, i].legend()
                axes[ch, i].grid()
        
        plt.tight_layout()
        self.logger.experiment.log({"Reconstruction Comparison": wandb.Image(fig)})
        plt.close(fig)
        self.input_signals = None
