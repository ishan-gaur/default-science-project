"""
This training script trains an augmented ESM model (predictor-tilted or LoRA-SFT'ed)
to generate sequences from the dataset. The models are trained with a cross-entropy loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
from sacred import Experiment
from sacred.observers import TinyDbObserver
from tqdm import tqdm
from torchmetrics.classification import MultilabelF1Score
from torch.utils.data import DataLoader, WeightedRandomSampler
from pdg.esm_ec.src.model import Classifier, ECAggregate
from pdg.esm_ec.src.train import multilevel_loss, level_4_loss, balanced_accuracies
from pdg.esm_ec.src.data import ECEmbeddingDataset
from pdg.esm_ec.constants import SWISSPROT_EC_FOLDER, OUTPUT_FOLDER, LEVEL_SETS, SWISSPROT_MASKED_DATASET_FOLDER, ModelType, CHECKPOINT_FOLDER
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
import os

ex = Experiment("train_class_conditional")
ex.observers.append(TinyDbObserver("experiments.db"))

@ex.config
def config():
    resume_hash = None

    train_split, test_split = "train", "test"

    epochs = 100
    batch_size = 15360
    lr = 0.001

    label_smoothing = True
    validation_steps = 100  # Run validation every N steps
    

@ex.automain
def main(
    resume_hash: Optional[str],
    train_split: str,
    test_split: str,
    epochs: int,
    batch_size: int,
    lr: float,
    label_smoothing: bool,
    validation_steps: int,
    _run,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(SWISSPROT_EC_FOLDER / LEVEL_SETS, "rb") as f:
        level_ec_to_idx = pkl.load(f)

    # Load data
    # TODO: choose dataset
    # train_dataset = ECEmbeddingDataset.from_file(SWISSPROT_MASKED_DATASET_FOLDER, split=train_split, model=ModelType.ESM2_650M, include_mask_level=True)
    # test_dataset = ECEmbeddingDataset.from_file(SWISSPROT_MASKED_DATASET_FOLDER, split=test_split, model=ModelType.ESM2_650M, include_mask_level=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    # TODO: choose model
    # model = ECAggregate(level_ec_to_idx)
    model.to(device)
    
    # Log total number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    _run.info["total_parameters"] = total_params
    print(f"Total model parameters: {total_params:,}")
    
    # Configure loss and optimizer
    # TODO: configure loss
    # criterion = nn.CrossEntropyLoss(label_smoothing=0.1 if label_smoothing else 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Setup logging, checkpointing, and early stopping
    best_loss = float('inf')
    step_count = 0
    start_epoch = 0
    
    # Create checkpoint directory
    checkpoint_dir = CHECKPOINT_FOLDER / f"run_{_run._id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_best_checkpoint(epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'step_count': step_count,
            'config': dict(_run.config)
        }
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        _run.add_artifact(str(best_path), name="best_model.pt")

    def validate():
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Validation"):
                # TODO: implement validation step
                pass
        return val_loss / len(test_loader)
    
    # Resume from checkpoint if specified
    if resume_hash:
        checkpoint_path = CHECKPOINT_FOLDER / f"run_{resume_hash}" / "best_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            step_count = checkpoint['step_count']
            print(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            # TODO: implement training step
            step_count += 1
            
            # Run validation every validation_steps
            if step_count % validation_steps == 0:
                val_loss = validate()
                print(f"Step {step_count}, Validation Loss: {val_loss:.4f}")
                
                # Log validation loss
                _run.log_scalar("val_loss", val_loss, step_count)
                _run.info["latest_val_loss"] = val_loss
                
                # Check if this is the best model and save checkpoint
                if val_loss < best_loss:
                    best_loss = val_loss
                    _run.info["best_val_loss"] = best_loss
                    save_best_checkpoint(epoch, val_loss)
            
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        
        # Log training loss per epoch
        _run.log_scalar("train_loss", train_loss, epoch + 1)
        _run.info["latest_train_loss"] = train_loss
    
    # Save final model as artifact
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(_run.config)
    }, final_path)
    _run.add_artifact(str(final_path), name="final_model.pt")