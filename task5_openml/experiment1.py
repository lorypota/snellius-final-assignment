import os
import torch
import torch.nn as nn
import torchmetrics
import openml
import openml_pytorch as opt
from torchvision import transforms
import argparse
import wandb
from pathlib import Path

# Import local modules
from .base_setup import get_openml_transform, get_data_dir

openml.config.server = 'https://api.openml.org/api/v1/'

class WandbCallback:
    def __init__(self, project_name, entity_name, config=None):
        self.project_name = project_name
        self.entity_name = entity_name
        self.config = config or {}
        self.initialized = False
        
    def initialize(self, num_classes):
        """Initialize W&B once we know the number of classes"""
        if not self.initialized:
            wandb.init(
                project=self.project_name, 
                entity=self.entity_name,
                config=dict(self.config, num_classes=num_classes)
            )
            self.initialized = True
            
    def on_epoch_end(self, epoch, train_metrics, valid_metrics):
        """Log metrics at the end of each epoch"""
        if not self.initialized:
            return
            
        # Log training metrics
        train_log = {f"train_{k}": v for k, v in train_metrics.items()}
        # Log validation metrics
        valid_log = {f"valid_{k}": v for k, v in valid_metrics.items()}
        # Log epoch
        combined_log = {**train_log, **valid_log, "epoch": epoch}
        wandb.log(combined_log)
        
    def on_fold_start(self, fold):
        """Log the current fold"""
        if not self.initialized:
            return
        wandb.log({"fold": fold})
        
    def on_fold_end(self, fold, metrics):
        """Log metrics at the end of a fold"""
        if not self.initialized:
            return
        fold_metrics = {f"fold_{fold}_{k}": v for k, v in metrics.items()}
        wandb.log(fold_metrics)
        
    def on_run_end(self, run):
        """Log final run information and finish W&B run"""
        if not self.initialized:
            return
        
        try:
            # Try to log fold evaluations if available
            if hasattr(run, 'fold_evaluations'):
                for metric, fold_values in run.fold_evaluations.items():
                    if metric == 'predictions':
                        continue  # Skip predictions, too large
                    for fold, value in fold_values.items():
                        wandb.log({f"final_{metric}_fold_{fold}": value})
                
            # Log URL of the run on OpenML
            if hasattr(run, 'run_id'):
                wandb.log({"openml_run_id": run.run_id})
                wandb.run.summary["openml_url"] = f"https://www.openml.org/r/{run.run_id}"
        except Exception as e:
            print(f"Warning: Error logging final metrics to W&B: {e}")
        
        wandb.finish()

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

class BirdClassifier1(nn.Module):
    def __init__(self, num_classes=315, dropout_rate=0.4):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution: 3x128x128 -> 64x128x128 (no downsampling here)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Create residual layers with ResNet-18 architecture (2,2,2,2)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 64 channels
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # 128 channels
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # 256 channels
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # 512 channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Final fully connected layer to map features to num_classes
        self.fc = nn.Linear(512, num_classes)
        
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
        # The first block in this layer might need downsampling
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        # Add remaining blocks
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution and activation
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pool and flatten
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Dropout
        x = self.dropout(x)
        
        # Final classification layer
        x = self.fc(x)
        
        # Apply softmax
        return torch.softmax(x, dim=1)
    
    def predict_proba(self, X):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Convert input to tensor if needed
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            
            # Forward pass (which already includes softmax)
            probabilities = self(X)
            
            # Return as numpy array as expected by OpenML
            return probabilities.cpu().numpy()

def evaluate_model_and_publish_results(model, trainer):
    task = openml.tasks.get_task(363465)   # Get data and crossvalidation splits
    num_classes = len(task.class_labels)   # Sets the correct number of classes
    
    # Create W&B callback
    wandb_cb = WandbCallback(
        project_name="birds_classification_openml",
        entity_name="lorypota-eindhoven-university-of-technology",
        config={
            "model": model.__name__,
            "dataset": "BirdsOpenML",
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
    )
    
    # Initialize W&B with num_classes
    wandb_cb.initialize(num_classes)
    
    # Add our custom callback to the trainer's callbacks
    if trainer.callbacks is None:
        trainer.callbacks = []
    trainer.callbacks.append(wandb_cb)
    
    # Initialize model
    model_instance = model(num_classes=num_classes)
    
    print("Training model...")
    try:
        # Train model with OpenML
        run = openml.runs.run_model_on_task(model_instance, task, avoid_duplicate_runs=True)
        
        # Save model to disk
        model_path = os.path.join(CHECKPOINTS_DIR, f"{model.__name__}.pt")
        torch.save(model_instance.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        print("Adding experiment info to run...")
        run = opt.add_experiment_info_to_run(run=run, trainer=trainer)
        
        # Let the callback know the run is ending
        wandb_cb.on_run_end(run)
        
        try:
            run.publish()
            print("Run is uploaded at https://www.openml.org/r/{}".format(run.run_id))
        except Exception as e:
            print(f"Error publishing to OpenML: {e}")
            print("Model was saved locally but couldn't be published to OpenML.")
    
    except Exception as e:
        print(f"Error during training or evaluation: {e}")
        wandb.finish()

# Define path constants
RESULTS_DIR = os.path.join(os.getcwd(), "results")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser(description='Train and evaluate bird classifier on OpenML')
parser.add_argument('--api_key', type=str, required=True, help='OpenML API key')
parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
args = parser.parse_args()

# Configure OpenML with API key
openml.config.apikey = args.api_key

# Create transform
transform = get_openml_transform()

# Create data module
data_module = opt.trainer.OpenMLDataModule(
    type_of_data="image",
    file_dir=get_data_dir(),
    filename_col="file_path",
    target_mode="categorical",
    target_column="CATEGORY",
    batch_size=args.batch_size,
    transform=transform
)

# Create trainer module
trainer = opt.trainer.OpenMLTrainerModule(
    experiment_name="Assignment-5, BirdClassifier1",
    data_module=data_module,
    verbose=True,
    epoch_count=args.epochs,
    metrics=[opt.metrics.accuracy],
    callbacks=[],
)

# Configure OpenML trainer
opt.config.trainer = trainer

# Evaluate and publish the model
evaluate_model_and_publish_results(model=BirdClassifier1, trainer=trainer)