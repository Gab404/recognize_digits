import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from cnn_net import MyNet

num_epochs = 4
learning_rate = 0.01
batch_size = 4
num_classes = 10
FILE = "save/my_model.pt"

# Transform each image into tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

# Set the training loader
train_data = datasets.MNIST('../data', train=True, download=True, transform=transform)
# Set the testing loader
test_data = datasets.MNIST('../data', train=False, download=True, transform=transform)

# put my data to batch (4 images in each batch)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

model = MyNet()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        #forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 2000 == 0:
            print(f'epoch {epoch + 1} / {num_epochs}, loss = {loss.item():.4f}')
print("End of training")

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')

# Save my model
torch.save(model.state_dict(), FILE)
print(f'Model saved with {acc} %')