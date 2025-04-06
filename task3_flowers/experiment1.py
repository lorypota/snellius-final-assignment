import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from core.logger import ModelLogger
from torch.utils.data import DataLoader
from core.fewshot import EpisodicSampler

from .base_setup import (
    train_dataset, val_dataset, test_dataset,
    train_sampler, val_sampler, test_sampler,
    idx_to_flower
)

# Define collate function
def episodic_collate(batch):
    return batch[0]  # Return first (and only) element of batch

# ---------------------------
# Prototypical Network Model
# ---------------------------
class ProtoNet(pl.LightningModule):
    def __init__(self, embedding_dim=512, lr=0.001):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lr = lr
        
        # ResNet18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        self.embedding = backbone
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.embedding(x)
    
    def compute_prototypes(self, support_x, support_y):
        support_embeddings = self(support_x)
        n_way = len(torch.unique(support_y))
        prototypes = torch.zeros(n_way, self.embedding_dim, device=self.device)
        
        for i in range(n_way):
            mask = support_y == i
            prototypes[i] = support_embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        query_expanded = query_embeddings.unsqueeze(1)
        prototypes_expanded = prototypes.unsqueeze(0)
        return ((query_expanded - prototypes_expanded) ** 2).sum(dim=2)
    
    def training_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        prototypes = self.compute_prototypes(support_x, support_y)
        query_embeddings = self(query_x)
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances
        
        loss = F.cross_entropy(logits, query_y)
        acc = self.train_acc(torch.argmax(logits, dim=1), query_y)
        
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        prototypes = self.compute_prototypes(support_x, support_y)
        query_embeddings = self(query_x)
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances
        
        loss = F.cross_entropy(logits, query_y)
        acc = self.val_acc(torch.argmax(logits, dim=1), query_y)
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        prototypes = self.compute_prototypes(support_x, support_y)
        query_embeddings = self(query_x)
        distances = self.compute_distances(query_embeddings, prototypes)
        logits = -distances
        
        acc = self.test_acc(torch.argmax(logits, dim=1), query_y)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        
        return acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

# ---------------------------
# Setup and Training
# ---------------------------
# cpus-per-task=9
num_workers = 9  

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

# Initialize model
model = ProtoNet(embedding_dim=512, lr=0.001)

# Setup WandB logging
wandb_logger = WandbLogger(
    project="flowers_few_shot", 
    entity="lorypota-eindhoven-university-of-technology"
)

# Define paths for model saving
RESULTS_DIR = os.path.join(os.getcwd(), "results")
OUTPUTS_DIR = os.path.join(RESULTS_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Initialize the custom ModelLogger
model_logger = ModelLogger(
    model_name="protonet",
    test_loader=test_loader,
    output_dir=OUTPUTS_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Log hyperparameters
wandb_logger.log_hyperparams({
    "model": "ProtoNet",
    "backbone": "ResNet18",
    "embedding_dim": 512,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "n_way": 5,
    "k_shot": 5,
    "query_samples": 5,
    "episodes_per_epoch": 100
})

# Training
trainer = pl.Trainer(
    max_epochs=20,
    logger=wandb_logger,
    accelerator="auto",
    callbacks=[model_logger],
    log_every_n_steps=10,
    enable_progress_bar=False
)

# Start training
trainer.fit(model, train_loader, val_loader)

# Test on test set
trainer.test(model, test_loader)

# Evaluate with different K-shot settings
print("Evaluating with different K-shot settings...")
model.eval()

# Test with different shot values
for k_shot in [1, 3, 5, 10]:
    # Create a new sampler for this K-shot setting
    test_sampler_k = EpisodicSampler(
        test_dataset, 
        episodes_per_epoch=50,
        N_way=5, 
        K_shot=k_shot, 
        Q_query=15
    )
    
    test_loader_k = DataLoader(
        test_dataset,
        batch_sampler=test_sampler_k,
        collate_fn=episodic_collate,
        num_workers=num_workers
    )
    
    # Test and log results
    results = trainer.test(model, test_loader_k)
    print(f"{k_shot}-shot accuracy: {results[0]['test_acc']:.4f}")
    wandb_logger.log_metrics({f"test_acc_{k_shot}_shot": results[0]['test_acc']})