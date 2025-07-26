import torch
import torch.nn as nn
import torch.optim as optim
from models.classifier import FishMaskClassifier
from torchvision import transforms
from dataset import FishMaskDataset
from torch.utils.data import DataLoader

#Prepare Data
transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = FishMaskDataset(
    image_dir='data_test/train',
    label_file='data_test/train_labels.csv',
    transform=transforms
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

val_dataset = FishMaskDataset(
    image_dir='data_test/val',
    label_file='data_test/val_labels.csv',
    transform=transforms
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2
)
    

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = FishMaskClassifier(num_classes=2).to(device)

# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 20
for epoch in range(num_epochs):
    # ------------------
    # TRAINING PHASE
    # ------------------
    running_loss = 0.0
    correct = 0
    total = 0
    
    model.train()  # Set model to training mode
    
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()   # Clear old gradients
        loss.backward()         # Compute gradients
        optimizer.step()        # Update weights
        
        # Lpogging
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Epoch statistics
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = 100 * correct / total
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    # ------------------
    # VALIDATION PHASE
    # ------------------
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    model.eval()  # Set model to evaluation mode (turns off dropout, batchnorm, etc.)
    
    with torch.no_grad():  # No need to compute gradients during validation
        
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_loss /= len(val_dataset)
    val_accuracy = 100 * val_correct / val_total

    # ------------------
    # LOGGING
    # ------------------
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
    print("-" * 50)
    
    
# Save the model
torch.save(model.state_dict(), 'fish_mask_classifier.pt')