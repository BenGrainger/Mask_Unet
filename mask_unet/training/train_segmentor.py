import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    running_loss = 0.0
    model.train()
    for i, (images, masks) in enumerate(loader):

        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


def evaluation(model, val_loader, criterion, DEVICE, epoch, num_epochs):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
                
        mean_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], mean Loss: {mean_val_loss:.4f}")
    return mean_val_loss

def save_checkpoint(model, model_save_path):
    torch.save(model.state_dict(), model_save_path)

def train_segmentor(model, optimizer, train_loader, val_loader, criterion, DEVICE, num_epochs, model_save_path):
    best_val_loss = float('inf') 
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        mean_val_loss = evaluation(model, val_loader, criterion, DEVICE, epoch, num_epochs)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_checkpoint(model, model_save_path)
            print(f"Saved model with validation loss: {mean_val_loss:.4f}")



    