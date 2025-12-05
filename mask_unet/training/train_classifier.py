import torch


def train_one_epoch(model, optimizer, train_loader, criterion, DEVICE):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())  
        loss.backward()
        optimizer.step()

def evaluation(model, val_loader, criterion, DEVICE, epoch, num_epochs):
    model.eval()

    with torch.no_grad():
        val_loss = 0
        correct_preds = 0
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).long()  # Convert to 0 or 1
            correct_preds += (predicted == labels).sum().item()

        mean_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / len(val_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], mean Loss: {mean_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
    return mean_val_loss

def save_checkpoint(model, model_save_path):
    torch.save(model.state_dict(), model_save_path) 



def train_classifier(model, optimizer, train_loader, val_loader, criterion, DEVICE, num_epochs, model_save_path):
    best_val_loss = float('inf') 
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, criterion, DEVICE)
        mean_val_loss = evaluation(model, val_loader, criterion, DEVICE, epoch, num_epochs)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            save_checkpoint(model, model_save_path) 
            print(f"Saved model with validation loss: {mean_val_loss:.4f}")

