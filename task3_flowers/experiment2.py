import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl
import wandb
import torchmetrics
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .base_setup import (
    train_sampler, val_sampler, test_sampler, idx_to_flower, EpisodicDataLoader,
    RESULTS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR
)
from core.logger import ModelLogger

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ResNet embedding network
class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_size=64):
        super(ResNetEmbedding, self).__init__()
        # Load pre-trained ResNet18
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add a new FC layer
        self.fc = nn.Linear(512, embedding_size)
        
        # Initialize the FC layer
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ProtoNet implementation as a PyTorch Lightning module
class ProtoNetLightning(pl.LightningModule):
    def __init__(self, embedding_model, learning_rate=0.0001, weight_decay=0.01):
        super(ProtoNetLightning, self).__init__()
        self.embedding = embedding_model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        
    def forward(self, support_x, support_y, query_x):
        # Extract features
        support_z = self.embedding(support_x)  # [N_way * K_shot, embedding_size]
        query_z = self.embedding(query_x)      # [N_way * Q_query, embedding_size]
        
        # Get unique class labels
        classes = torch.unique(support_y)
        n_classes = len(classes)
        
        # Compute prototypes
        prototypes = torch.zeros(n_classes, support_z.shape[1], device=self.device)
        for i, c in enumerate(classes):
            # Select embeddings of the same class and average them
            mask = support_y == c
            class_z = support_z[mask]
            prototypes[i] = class_z.mean(0)
        
        # Compute distances between query examples and prototypes
        dists = torch.cdist(query_z, prototypes)**2  # Squared Euclidean distance
        
        # Convert distances to log probabilities (negative distances)
        query_logits = -dists
        
        return query_logits
    
    def training_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Forward pass
        query_logits = self(support_x, support_y, query_x)
        
        # Compute loss and accuracy
        loss = F.cross_entropy(query_logits, query_y)
        preds = query_logits.argmax(dim=1)
        acc = self.train_accuracy(preds, query_y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Forward pass
        query_logits = self(support_x, support_y, query_x)
        
        # Compute loss and accuracy
        loss = F.cross_entropy(query_logits, query_y)
        preds = query_logits.argmax(dim=1)
        acc = self.val_accuracy(preds, query_y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Forward pass
        query_logits = self(support_x, support_y, query_x)
        
        # Compute accuracy
        preds = query_logits.argmax(dim=1)
        acc = self.test_accuracy(preds, query_y)
        
        # Log metrics
        self.log('test_acc', acc, on_epoch=True)
        
        return acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=20,  # max epochs
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }

# Create data module for PyTorch Lightning
class ProtoNetDataModule(pl.LightningDataModule):
    def __init__(self, train_sampler, val_sampler, test_sampler):
        super().__init__()
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
        
    def train_dataloader(self):
        return EpisodicDataLoader(self.train_sampler)
    
    def val_dataloader(self):
        return EpisodicDataLoader(self.val_sampler)
    
    def test_dataloader(self):
        return EpisodicDataLoader(self.test_sampler)

def main():
    # Initialize Weights & Biases
    wandb.init(
        project="flower-protonet", 
        name="resnet_protonet",
        entity="lorypota-eindhoven-university-of-technology"
    )
    
    # Hyperparameters
    hyperparams = {
        "embedding_size": 64,
        "learning_rate": 0.0001,  # Lower learning rate for pretrained model
        "weight_decay": 0.01,
        "max_epochs": 5,
        "architecture": "ResNet18",
        "N_way": 5,
        "K_shot": 5,
        "scheduler": "CosineAnnealing",
        "pretrained": True
    }
    
    # Create embedding model and ProtoNet
    embedding_model = ResNetEmbedding(embedding_size=hyperparams["embedding_size"])
    protonet = ProtoNetLightning(
        embedding_model,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"]
    )
    
    # Create data module
    data_module = ProtoNetDataModule(train_sampler, val_sampler, test_sampler)
    
    # Create loggers
    wandb_logger = WandbLogger(
        project="flower-protonet",
        entity="lorypota-eindhoven-university-of-technology"
    )
    wandb_logger.log_hyperparams(hyperparams)
    
    model_logger = ModelLogger(
        model_name="resnet_protonet",
        test_loader=None,
        output_dir=OUTPUTS_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        filename='protonet-resnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    # Train the model
    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"],
        callbacks=[checkpoint_callback, model_logger],
        logger=wandb_logger,
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,
        gradient_clip_val=0.5,
        enable_progress_bar=False
    )
    
    trainer.fit(protonet, data_module)
    
    # Test the model
    test_results = trainer.test(protonet, data_module)
    print(f"Test Results: {test_results}")
    
    # Log test results to W&B
    wandb.log({"test_accuracy": test_results[0]["test_acc"]})
    
    # Close W&B
    wandb.finish()

if __name__ == "__main__":
    main()