import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from core.logger import ModelLogger
import multiprocessing

# Import the base setup
from .base_setup import (
    flower_ds, train_dataset, val_dataset, test_dataset,
    train_sampler, val_sampler, test_sampler,
    flower_to_idx, idx_to_flower, EpisodicSampler
)

# ---------------------------
# Model definitions for Task 2
# ---------------------------

# Simple CNN backbone for embedding
class EmbeddingCNN(nn.Module):
    def __init__(self, in_channels=3, embedding_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 64 x 64 x 64
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 32 x 32 x 128
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 16 x 16 x 256
        
        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)  # 8 x 8 x 512
        
        # Global pooling and projection
        x = self.adaptive_pool(x)  # 1 x 1 x 512
        x = x.view(x.size(0), -1)  # Flatten: 512
        x = self.fc(x)  # embedding_dim
        
        return x

# Alternative: Use a pretrained model
class PretrainedEmbedding(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        # Use a smaller model like MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=True)
        # Replace the classifier with a new projection layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, embedding_dim)
        )
        
    def forward(self, x):
        return self.backbone(x)

# Prototypical Network implementation
class ProtoNet(pl.LightningModule):
    def __init__(self, embedding_network="simple", embedding_dim=512, lr=0.001, distance="euclidean"):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.distance_type = distance
        self.lr = lr
        
        # Select embedding network
        if embedding_network == "simple":
            self.embedding = EmbeddingCNN(embedding_dim=embedding_dim)
        elif embedding_network == "mobilenet":
            self.embedding = PretrainedEmbedding(embedding_dim=embedding_dim)
        elif embedding_network == "resnet18":
            # Use ResNet18 as backbone
            backbone = models.resnet18(pretrained=True)
            # Replace FC layer with identity to get embeddings
            backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
            self.embedding = backbone
        else:
            raise ValueError(f"Unknown embedding network: {embedding_network}")
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)  # 5-way classification
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.embedding(x)
    
    def compute_prototypes(self, support_x, support_y, n_way):
        """Compute class prototypes from support examples."""
        # Get embeddings for support examples
        support_embeddings = self(support_x)  # [n_support, embedding_dim]
        
        # Initialize prototypes tensor
        prototypes = torch.zeros(n_way, self.embedding_dim, device=support_embeddings.device)
        
        # For each class, compute the mean of support embeddings
        for i in range(n_way):
            # Get indices of examples belonging to class i
            mask = support_y == i
            # Compute mean embedding for class i
            if mask.sum() > 0:
                prototypes[i] = support_embeddings[mask].mean(dim=0)
        
        return prototypes
    
    def compute_distances(self, query_embeddings, prototypes):
        """Compute distances from query embeddings to prototypes."""
        # Calculate Euclidean distance between queries and prototypes
        n_queries = query_embeddings.shape[0]
        n_prototypes = prototypes.shape[0]
        
        if self.distance_type == "euclidean":
            # Reshape for broadcasting
            queries_expanded = query_embeddings.unsqueeze(1)  # [n_queries, 1, embedding_dim]
            prototypes_expanded = prototypes.unsqueeze(0)      # [1, n_prototypes, embedding_dim]
            
            # Euclidean distance (squared)
            distances = ((queries_expanded - prototypes_expanded) ** 2).sum(dim=2)  # [n_queries, n_prototypes]
            return distances
        elif self.distance_type == "cosine":
            # Normalize embeddings for cosine similarity
            query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=1)
            prototypes_norm = F.normalize(prototypes, p=2, dim=1)
            
            # Compute cosine similarity
            similarity = torch.mm(query_embeddings_norm, prototypes_norm.t())  # [n_queries, n_prototypes]
            
            # Convert to distance (1 - similarity)
            distances = 1 - similarity
            return distances
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
    
    def training_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Convert lists to tensors if needed
        if isinstance(support_x, list):
            support_x = torch.stack(support_x)
        if isinstance(support_y, list):
            support_y = torch.tensor(support_y, device=self.device)
        if isinstance(query_x, list):
            query_x = torch.stack(query_x)
        if isinstance(query_y, list):
            query_y = torch.tensor(query_y, device=self.device)
        
        # Determine N-way from the data
        n_way = len(torch.unique(support_y))
        
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        
        # Get embeddings for query set
        query_embeddings = self(query_x)
        
        # Compute distances from queries to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        # Compute loss
        loss = F.cross_entropy(logits, query_y)
        
        # Compute accuracy
        accuracy = self.train_acc(torch.argmax(logits, dim=1), query_y)
        
        # Log metrics
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Convert lists to tensors if needed
        if isinstance(support_x, list):
            support_x = torch.stack(support_x)
        if isinstance(support_y, list):
            support_y = torch.tensor(support_y, device=self.device)
        if isinstance(query_x, list):
            query_x = torch.stack(query_x)
        if isinstance(query_y, list):
            query_y = torch.tensor(query_y, device=self.device)
        
        # Determine N-way from the data
        n_way = len(torch.unique(support_y))
        
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        
        # Get embeddings for query set
        query_embeddings = self(query_x)
        
        # Compute distances from queries to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        # Compute loss
        loss = F.cross_entropy(logits, query_y)
        
        # Compute accuracy
        accuracy = self.val_acc(torch.argmax(logits, dim=1), query_y)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Convert lists to tensors if needed
        if isinstance(support_x, list):
            support_x = torch.stack(support_x)
        if isinstance(support_y, list):
            support_y = torch.tensor(support_y, device=self.device)
        if isinstance(query_x, list):
            query_x = torch.stack(query_x)
        if isinstance(query_y, list):
            query_y = torch.tensor(query_y, device=self.device)
        
        # Determine N-way from the data
        n_way = len(torch.unique(support_y))
        
        # Compute prototypes from support set
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        
        # Get embeddings for query set
        query_embeddings = self(query_x)
        
        # Compute distances from queries to prototypes
        distances = self.compute_distances(query_embeddings, prototypes)
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        # Compute accuracy
        accuracy = self.test_acc(torch.argmax(logits, dim=1), query_y)
        
        # Log metrics
        self.log("test_acc", accuracy, prog_bar=True, on_epoch=True)
        
        return accuracy
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch"
            }
        }

# Custom collate function for episodic batches
def episodic_collate(batch):
    # Each batch is already an episode from EpisodicSampler
    # Just return the first (and only) element of the batch
    return batch[0]

# ---------------------------
# Instantiate model and loggers
# ---------------------------
model_1 = ProtoNet(
    embedding_network="resnet18",  # Use ResNet18 as backbone
    embedding_dim=512,
    lr=0.001,
    distance="euclidean"
)

# Initialize the WandbLogger
wandb_logger = WandbLogger(
    project="flowers_few_shot", 
    entity="lorypota-eindhoven-university-of-technology"
)

# Define paths for model saving
RESULTS_DIR = os.path.join(os.getcwd(), "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
OUTPUTS_DIR = os.path.join(RESULTS_DIR, "outputs")

# Create directories if they don't exist
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Initialize the custom ModelLogger with a persistent path
model_logger = ModelLogger(
    model_name="protonet",
    test_loader=None,  # We'll use our own test function
    output_dir=OUTPUTS_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create DataLoaders for episodic training
num_workers = 9

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_sampler=test_sampler,
    collate_fn=episodic_collate,
    num_workers=num_workers
)

# Log hyperparameters
wandb_logger.log_hyperparams({
    "embedding_network": "resnet18",
    "embedding_dim": 512,
    "learning_rate": 0.001,
    "weight_decay": 1e-5,
    "n_way": 5,
    "k_shot": 5,
    "query_samples": 5,
    "distance_metric": "euclidean",
    "episodes_per_epoch": 100
})

# ---------------------------
# Training configuration
# ---------------------------
def train_model(model, max_epochs=20, resume_checkpoint=None):
    # Add early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )
    
    # Add model checkpoint to save to results/checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINTS_DIR,
        monitor='val_loss',
        filename='protonet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback, model_logger],
        gradient_clip_val=1.0,  # Add gradient clipping
        enable_progress_bar=False,
    )
    
    if resume_checkpoint:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)
    
    # Test the model on the test set
    trainer.test(model, test_loader)
    
    return model

# ---------------------------
# Run training or load model
# ---------------------------
train_model(model_1, max_epochs=30)

# ---------------------------
# Evaluate model on test set for different K-shot settings
# ---------------------------
def evaluate_different_shots(model, test_dataset, shots_to_test=[1, 3, 5, 10]):
    """Evaluate the model on different K-shot settings."""
    results = {}
    
    for k_shot in shots_to_test:
        # Create sampler and loader for this K-shot setting
        test_sampler_k = EpisodicSampler(
            test_dataset, 
            episodes_per_epoch=50,  # Use fewer episodes for evaluation
            N_way=5, 
            K_shot=k_shot, 
            Q_query=15  # Use more query examples for better evaluation
        )
        
        test_loader_k = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=test_sampler_k,
            collate_fn=episodic_collate,
            num_workers=num_workers
        )
        
        # Test the model
        trainer = pl.Trainer(
            accelerator="auto",
            logger=wandb_logger,
            enable_progress_bar=False
        )
        
        test_results = trainer.test(model, test_loader_k)
        results[k_shot] = test_results[0]["test_acc"]
        
        print(f"{k_shot}-shot accuracy: {results[k_shot]:.4f}")
    
    return results

# Run evaluation after training
print("Evaluating model on different K-shot settings...")
model_1.eval()
shots_results = evaluate_different_shots(model_1, test_dataset, shots_to_test=[1, 3, 5, 10])

# Log results to wandb
for k_shot, acc in shots_results.items():
    wandb_logger.log_metrics({f"test_acc_{k_shot}_shot": acc})