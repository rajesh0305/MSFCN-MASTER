import torch
from torch.utils.data import DataLoader, TensorDataset
from MSFCN2D import MSFCN2D  # Import your model

# Step 1: Load the test dataset (example with random data)
batch_size = 8
time_num = 4
band_num = 4
class_num = 4
height, width = 64, 64

# Random test dataset (replace with your actual test dataset)
x_test = torch.randn(20, time_num * band_num, height, width)  # 20 test samples
y_test = torch.randint(0, class_num, (20,))  # 20 test labels

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MSFCN2D(time_num, band_num, class_num).to(device)
model.load_state_dict(torch.load("MSFCN2D_model.pth"))
model.eval()

# Step 3: Evaluate the model
def evaluate_model(model, dataloader, device):
    correct = 0
    total = 0
    all_predictions = []  # To store all predictions
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Store predictions
            all_predictions.extend(predicted.cpu().numpy())

            # Calculate accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return accuracy, all_predictions

# Step 4: Run evaluation
test_accuracy, test_predictions = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Predictions: {test_predictions}")
