import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Import the base setup
from .base_setup import train_loader, val_loader, test_loader, label_to_idx, idx_to_label

# ---------------------------
# Model definitions for Task 1
# ---------------------------

# Basic Residual Block (as in ResNet)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First conv layer (with potential downsampling)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Downsample if input shape doesn't match output shape
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x  # Save input for skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

# ResNet‑18–style network built from scratch.
class birdsCNN(pl.LightningModule):
    def __init__(self, block, layers, num_classes=315, dropout_rate=0.3):
        super().__init__()
        self.in_channels = 64
        # Initial convolution: 3x128x128 -> 64x128x128 (no downsampling here)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Create residual layers. Note: stride 2 downsamples.
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # 64 channels
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)   # 128 channels
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)   # 256 channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)   # 512 channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Final fully connected layer to map features to num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming/He initialization for Conv layers with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # Standard initialization for BatchNorm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Xavier/Glorot initialization for Linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        # The first block in this layer might need downsampling.
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        # Add remaining blocks.
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution and activation.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Residual layers.
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # Pool and flatten.
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Dropout
        x = self.dropout(x)
        # Final classification layer.
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B, 3, 128, 128], y: [B]
        y = y.long()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1), y)
        f1 = self.f1_score(torch.argmax(preds, dim=1), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True)
        self.log("train_f1", f1, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.accuracy(torch.argmax(preds, dim=1), y)
        f1 = self.f1_score(torch.argmax(preds, dim=1), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_f1", f1, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Increase weight decay for more regularization
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.02)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# ---------------------------
# Instantiate model and logger
# ---------------------------
# This configuration [2,2,2,2] corresponds to ResNet‑18.
model_1 = birdsCNN(BasicBlock, [2, 2, 2, 2], num_classes=315, dropout_rate=0.3)

# Initialize the WandbLogger.
wandb_logger = WandbLogger(
    project="birds_classification", 
    entity="lorypota-eindhoven-university-of-technology"
)
# Log hyperparameters.
wandb_logger.log_hyperparams({
    "learning_rate": 0.001,
    "batch_size": 32,
    "architecture": "ResNetModified1",
    "num_classes": 315,
    "max_epochs": 20,
    "dropout_rate": 0.3,
    "weight_decay": 0.02,
})

# ---------------------------
# Training configuration
# ---------------------------
def train_model(model, max_epochs=30, resume_checkpoint=None):
    # Add early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=True,
        mode='min'
    )
    
    # Add model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    
    if resume_checkpoint:
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_checkpoint)
    else:
        trainer.fit(model, train_loader, val_loader)

# ---------------------------
# Run training or load model
# ---------------------------
train_model(model_1, max_epochs=20)
