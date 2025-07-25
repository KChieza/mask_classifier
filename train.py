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

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = FishMaskClassifier(num_classes=2).to(device)

# Initialize loss function
criterion = nn.CrossEntropyLoss()

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
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
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

torch.save(model.state_dict(), 'fish_mask_classifier.pt')