import torch, random
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Transform each image into tensor ( numpy array => Tensor )
# Normalize data to win time in gradient descent
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

# define batch size to regroup all images into slot of 64 images
batch_size = 64

# set train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size)
# set test loader
test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False, download=True, transform=transform), batch_size=batch_size)

# set random weight for prediction
weight = torch.randn(784, 10, requires_grad=True)

def test(weights, test_loader):
    test_size = len(test_loader.dataset)
    correct = 0

    for batch_idx, (data, target) in enumerate(test_loader):
        # reshape each image, to follow generate activation value
        data = data.view((-1, 28*28))
        
        # generate activation value for each neuron
        outputs = torch.matmul(data, weights)
        
        # Transform value between 0 and 1
        softmax = F.softmax(outputs, dim=1)
        
        # Get the highest value / prediction
        pred = softmax.argmax(dim=1, keepdim=True)
        
        # calculate the number of good predictions to make an average
        n_correct = pred.eq(target.view_as(pred)).sum().item()
        correct += n_correct

    acc = correct / test_size # make average
    print(" Accuracy on test set", acc)
    return

# train AI

def train_ai(weight, train_loader):
    it = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Reset my weight
        if weight.grad is not None:
            weight.grad.zero_()

        # reshape each image, to follow generate activation value
        data = data.view((-1, 28*28))
        
        # Get output of 10 neuron / prediction
        outputs = torch.matmul(data, weight)

        # Calculate loss
        log_softmax = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(log_softmax, targets)
        print("\rLoss shape: {}".format(loss), end="")

        # Backtracking, to minimize error, compute gradient for each variable
        loss.backward()

        with torch.no_grad():
            weight -= 0.1*weight.grad

        it += 1
        if it % 100 == 0:
            # Check program evolution, print loss
            test(weight, test_loader)

        if it > 5000:
            break
            
train_ai(weight, train_loader)

# Get test data
batch_idx, (data, target) = next(enumerate(test_loader))
data = data.view((-1, 28*28))

outputs = torch.matmul(data, weight) # Get result for each neuron
softmax = F.softmax(outputs, dim=1) # Put each value between 0 and 1
pred = softmax.argmax(dim=1, keepdim=True) # Take the biggest value => prediction

random.seed()
random_index = random.randint(0, 63)
plt.imshow(data[random_index].view(28, 28), cmap="gray")
print("\n\nThe number is", int(pred[random_index]))
print("Expected:", int(target[random_index]))
plt.show()