import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from MSFCN2D import MSFCN2D  # Import your model
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Step 1: Prepare the dataset (example with random data)
batch_size = 8
time_num = 4
band_num = 4
class_num = 4
height, width = 64, 64

# Random dataset (replace with your actual dataset)
x_data = torch.randn(100, time_num * band_num, height, width)  # 100 samples
y_data = torch.randint(0, class_num, (100,))  # 100 labels

dataset = TensorDataset(x_data, y_data)

# Step 1: Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Debugging: Check dataset sizes
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Step 2: Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSFCN2D(time_num, band_num, class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 2: Add validation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            print(f"Model output shape: {outputs.shape}")  # Debugging: Check output shape
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    print(f"Validation Loss: {avg_loss}, Validation Accuracy: {accuracy}")  # Debugging
    return avg_loss, accuracy

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_accuracies = []

# Step 3: Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Evaluate on validation set
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

    # Log metrics
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {train_losses[-1]:.4f}, "
          f"Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

# Save the trained model
torch.save(model.state_dict(), "MSFCN2D_model.pth")

# Plot performance metrics
plt.figure(figsize=(10, 5))

# Plot training and validation loss
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

# Save and show the plot
plt.tight_layout()
plt.savefig("performance_metrics.png")
plt.show()
