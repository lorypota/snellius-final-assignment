import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
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
# MAML Model with 4-layer CNN
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class MAML(pl.LightningModule):
    def __init__(self, 
                 inner_lr=0.01, 
                 outer_lr=0.001, 
                 num_inner_steps=5):
        super().__init__()
        
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        
        # Use a simple CNN for MAML to keep computation manageable
        self.net = SimpleCNN(num_classes=5)
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=5)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.net(x)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.outer_lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Meta-training
        task_losses = []
        task_accs = []
        
        # Manual implementation of differentiable optimization
        # 1. Create a clone of the model with a copy of parameters
        fast_weights = [p.clone().detach().requires_grad_(True) for p in self.net.parameters()]
        
        # 2. Inner loop: adapt fast_weights on support set
        for _ in range(self.num_inner_steps):
            # Forward pass with the current fast weights
            support_logits = self.forward_with_weights(self.net, fast_weights, support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            # Compute gradients with respect to fast_weights
            grads = torch.autograd.grad(support_loss, fast_weights, create_graph=True)
            
            # Update fast_weights manually
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]
        
        # 3. Evaluate the adapted model on query set (outer loop)
        query_logits = self.forward_with_weights(self.net, fast_weights, query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = self.train_acc(torch.argmax(query_logits, dim=1), query_y)
        
        task_losses.append(query_loss)
        task_accs.append(query_acc)
        
        # Average losses and accuracies across tasks
        mean_loss = torch.mean(torch.stack(task_losses))
        mean_acc = torch.mean(torch.stack(task_accs))
        
        self.log("train_loss", mean_loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", mean_acc, prog_bar=True, on_epoch=True)
        
        return mean_loss
    
    def forward_with_weights(self, model, weights, x):
        """Forward pass using the provided weights instead of model.parameters()"""
        # Get a list of modules containing parameters
        module_dicts = [m for m in model.modules() if len(list(m.parameters())) > 0]
        
        # Cache the original parameters
        orig_params = []
        for module in module_dicts:
            orig_params.append([p.clone() for p in module.parameters()])
        
        # Assign the new weights to the model
        param_index = 0
        for module in module_dicts:
            for param_name, _ in module.named_parameters():
                module._parameters[param_name] = weights[param_index]
                param_index += 1
        
        # Forward pass
        output = model(x)
        
        # Restore the original parameters
        param_index = 0
        for i, module in enumerate(module_dicts):
            for j, (param_name, _) in enumerate(module.named_parameters()):
                module._parameters[param_name] = orig_params[i][j]
        
        return output
    
    def validation_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Create a clone of the model for adaptation
        val_net = SimpleCNN(num_classes=5).to(self.device)
        val_net.load_state_dict(self.net.state_dict())
        
        # Inner loop: adapt using support set
        optimizer = torch.optim.SGD(val_net.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            support_logits = val_net(support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            optimizer.zero_grad()
            support_loss.backward()
            optimizer.step()
        
        # Outer loop: evaluate on query set
        query_logits = val_net(query_x)
        query_loss = F.cross_entropy(query_logits, query_y)
        query_acc = self.val_acc(torch.argmax(query_logits, dim=1), query_y)
        
        self.log("val_loss", query_loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", query_acc, prog_bar=True, on_epoch=True)
        
        return query_loss
    
    def test_step(self, batch, batch_idx):
        support_x, support_y, query_x, query_y, _ = batch
        
        # Create a clone of the model for adaptation
        test_net = SimpleCNN(num_classes=5).to(self.device)
        test_net.load_state_dict(self.net.state_dict())
        
        # Inner loop: adapt using support set
        optimizer = torch.optim.SGD(test_net.parameters(), lr=self.inner_lr)
        
        for _ in range(self.num_inner_steps):
            support_logits = test_net(support_x)
            support_loss = F.cross_entropy(support_logits, support_y)
            
            optimizer.zero_grad()
            support_loss.backward()
            optimizer.step()
        
        # Evaluate on query set
        query_logits = test_net(query_x)
        query_acc = self.test_acc(torch.argmax(query_logits, dim=1), query_y)
        
        self.log("test_acc", query_acc, prog_bar=True, on_epoch=True)
        
        return query_acc

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
model = MAML(inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)

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
    model_name="maml",
    test_loader=test_loader,
    output_dir=OUTPUTS_DIR,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Log hyperparameters
wandb_logger.log_hyperparams({
    "model": "MAML",
    "backbone": "SimpleCNN",
    "inner_learning_rate": 0.01,
    "outer_learning_rate": 0.001,
    "inner_steps": 5,
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