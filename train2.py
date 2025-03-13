from util import *
import sys
import time
from train import *
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train2(model, device, train_dataloader, val_dataloader, config):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    train_losses = []
    val_losses = []

    for epoch in range(config["epochs"]):
        total_loss = 0
        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Initialize hidden state
            hidden = None
            
            # Teacher forcing
            loss = 0
            for t in range(inputs.size(1)):
                output, hidden = model(inputs[:, t].unsqueeze(1), hidden)
                loss += criterion(output.view(-1, model.vocab_size), targets[:, t])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % config["log_interval"] == 0:
                pbar.set_postfix(loss=loss.item() / inputs.size(1))

        avg_loss = total_loss / len(train_dataloader)
        train_loss = eval(model, device, val_dataloader=train_dataloader)
        val_loss = eval(model, device, val_dataloader=val_dataloader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch [{epoch+1}/{config['epochs']}] | Average Loss: {avg_loss:.4f} | Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")
        model.train()

    print("Training complete!")
    fname = config["path"]
    if config["save_path"]:
        os.makedirs(config["save_path"], exist_ok=True)
        torch.save(model, os.path.join(config['save_path'], fname))
        print("Model saved")
    plot_losses(train_losses, val_losses, fname=fname)