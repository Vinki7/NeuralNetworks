import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import config as cfg

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Input to first hidden layer
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()

        # First hidden to second hidden layer
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        # Second hidden to output layer
        self.fc3 = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)  # Output activation for classification

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image (from 28x28 to 1D)
        x = self.relu1(self.fc1(x))  # First hidden layer with ReLU
        x = self.relu2(self.fc2(x))  # Second hidden layer with ReLU
        x = self.softmax(self.fc3(x))  # Output layer with LogSoftmax
        return x


def train_model(model, optimizer, train_loader, test_loader, epochs=cfg.EPOCH_COUNT):
    device = cfg.DEVICE if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses, accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_loader))

        model.eval()
        correct, total, test_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_losses.append(test_loss / len(test_loader))
        accuracy = 100 * correct / total
        accuracies.append(accuracy)

        print(
            f'Epoch {epoch + 1}/{epochs}, Cumulative loss: {running_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return model, train_losses, test_losses, accuracies


def evaluate_model(model, test_loader):
    model.eval()
    device = cfg.DEVICE if torch.cuda.is_available() else 'cpu'  # Ensure consistent device handling
    model.to(device)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, optimizer_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[i for i in range(10)])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {optimizer_name}')
    plt.show()


def plot_training_results(train_losses, test_losses, accuracies, optimizer_name):
    epochs = range(1, len(train_losses) + 1)

    # Plot Training and Test Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Test Loss - {optimizer_name}')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Accuracy - {optimizer_name}')
    plt.legend()

    plt.show()


def run(sgd_rate: float, sgd_mom_rate: float, momentum: float, adam_rate: float):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train with SGD
    print("\nTraining with SGD:")
    model_sgd = MNISTModel()
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=sgd_rate)
    model_sgd, train_losses_sgd, test_losses_sgd, accuracies_sgd = train_model(model_sgd, optimizer_sgd, train_loader, test_loader)
    plot_training_results(train_losses_sgd, test_losses_sgd, accuracies_sgd, "SGD")
    y_true_sgd, y_pred_sgd = evaluate_model(model_sgd, test_loader)
    plot_confusion_matrix(y_true_sgd, y_pred_sgd, "SGD")

    # Train with SGD + Momentum
    print("\nTraining with SGD + Momentum:")
    model_momentum = MNISTModel()
    optimizer_momentum = optim.SGD(model_momentum.parameters(), lr=sgd_mom_rate, momentum=momentum)
    model_momentum, train_losses_momentum, test_losses_momentum, accuracies_momentum = train_model(model_momentum, optimizer_momentum, train_loader, test_loader)
    plot_training_results(train_losses_momentum, test_losses_momentum, accuracies_momentum, "SGD + Momentum")
    y_true_momentum, y_pred_momentum = evaluate_model(model_momentum, test_loader)
    plot_confusion_matrix(y_true_momentum, y_pred_momentum, "SGD + Momentum")

    # Train with Adam
    print("\nTraining with Adam:")
    model_adam = MNISTModel()
    optimizer_adam = optim.Adam(model_adam.parameters(), lr=adam_rate)
    model_adam, train_losses_adam, test_losses_adam, accuracies_adam = train_model(model_adam, optimizer_adam, train_loader, test_loader)
    plot_training_results(train_losses_adam, test_losses_adam, accuracies_adam, "Adam")
    y_true_adam, y_pred_adam = evaluate_model(model_adam, test_loader)
    plot_confusion_matrix(y_true_adam, y_pred_adam, "Adam")


if __name__ == "__main__":
    # Example of running the function with learning rates and momentum
    run(cfg.SGD_RATE, cfg.SGD_MOMENTUM_RATE, cfg.MOMENTUM, cfg.ADAM_RATE)


