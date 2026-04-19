import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

batch_size = 64
lr = 0.001
epochs = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*14*14, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16*14*14)
        x = self.fc1(x)
        return x

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    loss_sum = 0
    for data, label in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    print(f"Epoch {epoch+1}, Loss: {loss_sum/len(train_loader)}")

correct = 0
total = 0
with torch.no_grad():
    for data, label in test_loader:
        out = model(data)
        _, pred = torch.max(out, 1)
        total += label.size(0)
        correct += (pred == label).sum().item()

print(f"准确率: {100*correct/total}%")
