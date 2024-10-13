import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
#from tqdm.notebook import tqdm

# %matplotlib inline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use ImageNet normalization
])

train_dataset = torchvision.datasets.ImageFolder("./data/train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder("./data/test", transform=transform)

#train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])
# Calculate the split lengths
train_size = int(0.9 * len(train_dataset))  # 90% for training
val_size = len(train_dataset) - train_size  # Remaining for validation (ensures correct sum)

# Split the dataset
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 30

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)



# These are what the class labels in the Fractions dataset represent. For more information,
# visit https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images?resource=download
classes = ["Fractured", "Not Fractured"]

def model_fully_connected(M) -> nn.Module:
    
    """Instantiate a linear model and send it to device."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, M), # Input size is 224x224x3
        nn.ReLU(),
        nn.Linear(M, 2) # Output size is 2 (binary classification)
    )
    return model.to(DEVICE)

'''def model_convolutional(M, k, N) -> nn.Module:
    # Model with a convolutional layer, ReLU, max-pooling, and a fully connected output
    model = nn.Sequential(
        nn.Conv2d(3, M, k),
        nn.ReLU(),
        nn.MaxPool2d(N,N),
        nn.Flatten(),
        nn.Linear(((33 - k) // N) * ((33 - k) // N) * M, 10)
    )
    return model.to(DEVICE)
'''
def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
    )-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for e in range(epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        print(f"epoch {e}")
        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))
    
    
    return train_losses, train_accuracies, val_losses, val_accuracies

def parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    (modifided to search over and return
     other parameters beyond learning rate.)
    """
    num_iter = 20
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    best_m = 0.0
    Ms = np.array([128, 256, 512, 750, 880, 960, 1024])
    lrs = np.linspace(1e-4, 0.01, 10)
    for _ in range(num_iter):
        m = np.random.choice(Ms)
        lr = np.random.choice(lrs)
        momentum=0.9
        print(f"trying learning rate {lr} and M={m}")
        model = model_fn(M=m)
        optim = SGD(model.parameters(), lr=lr, momentum=momentum)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr
            best_m = m
        print(f"lr={lr}, M={m} - Validation Accuracy: {max(val_acc)}")
    

    return best_lr, best_m


'''
3b
'''
'''def parameter_search(train_loader: DataLoader, val_loader: DataLoader, model_fn: Callable[..., nn.Module]) -> Tuple[float, int, int, int]:
    """
    Parameter search for the convolutional model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called with hyperparameters, returns a torch.nn.Module.

    Returns:
    The best hyperparameters with the least validation loss.
    """
    num_iter = 20
    best_val_accuracy = 0
    best_params = {"M": 0, "k": 0, "N": 0, "lr": 0.0, "momentum": 0.0}

    Ms = [100, 150, 200, 250, 300]  # Example filter counts
    ks = [3, 4, 5, 6]  # Example kernel sizes
    Ns = [3, 4]  # Example pool sizes
    lrs = np.linspace(1e-4, 0.01, 10)  # Learning rates
    momentums = [0.93, 0.95]  # Momentums

    for e in range(num_iter):
        print(f"iter {e}")
        M = np.random.choice(Ms)
        k = np.random.choice(ks)
        N = np.random.choice(Ns)
        lr = np.random.choice(lrs)
        momentum = np.random.choice(momentums)
        print(f"Trying M={M}, k={k}, N={N}, lr={lr}, momentum={momentum}")
        
        model = model_fn(M, k, N)
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
        
        _, _, _, val_accuracies = train(model, optimizer, train_loader, val_loader, epochs=20)
        print(f"hyperparamter Validation accuracy: {max(val_accuracies)}")
        if max(val_accuracies) > best_val_accuracy:
            best_val_accuracy = max(val_accuracies)
            best_params = {"M": M, "k": k, "N": N, "lr": lr, "momentum": momentum}
            #print(f"New best hyperparameters found: {best_params}, Validation Accuracy: {best_val_accuracy}")
        
    print(f"exiting hyperparamter search")
    return best_params["M"], best_params["k"], best_params["N"], best_params["lr"], best_params["momentum"]
'''
"""evaluate model on the testing data."""
def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc


#best_lr,best_m = parameter_search(train_loader, val_loader, model_fully_connected)
model1 = model_fully_connected(960)
optimizer1 = SGD(model1.parameters(), 0.00334, momentum=0.9)

model2 = model_fully_connected(256)
optimizer2 = SGD(model2.parameters(), 0.00556, momentum=0.9)

model3 = model_fully_connected(750)
optimizer3 = SGD(model3.parameters(), 0.0045, momentum=0.9)


train_loss, train_accuracy1, val_loss, val_accuracy1 = train(
    model1, optimizer1, train_loader, val_loader, 30)
test_loss, test_acc1 = evaluate(model1, test_loader)
print(f"Model 1's Test Accuracy: {test_acc1}, Validation Accuracy {max(val_accuracy1)}")

train_loss, train_accuracy2, val_loss, val_accuracy2 = train(
    model2, optimizer2, train_loader, val_loader, 30)
test_loss, test_acc2 = evaluate(model2, test_loader)
print(f"Model 2's Test Accuracy: {test_acc2}, Validation Accuracy {max(val_accuracy2)}")

train_loss, train_accuracy3, val_loss, val_accuracy3 = train(
    model3, optimizer3, train_loader, val_loader, 30)
test_loss, test_acc3 = evaluate(model3, test_loader)
print(f"Model 3's Test Accuracy: {test_acc3}, Validation Accuracy {max(val_accuracy3)}")


epochs = range(1, 31)

plt.plot(epochs, train_accuracy1, label="Model 1 Train Accuracy with m=960, lr=0.00334", linestyle='solid', color='blue')
plt.plot(epochs, val_accuracy1, label="Model 1 Validation Accuracy with m=960, lr=0.00334", linestyle='--', color='blue')

plt.plot(epochs, train_accuracy2, label="Model 2 Train Accuracy with m=256, lr=0.00556", linestyle='solid', color='green')
plt.plot(epochs, val_accuracy2, label="Model 2 Validation Accuracy with m=256, lr=0.00556", linestyle='--', color='green')

plt.plot(epochs, train_accuracy3, label="Model 3 Train Accuracy with m=750, lr=0.00445", linestyle='solid', color='purple')
plt.plot(epochs, val_accuracy3, label="Model 3 Validation Accuracy with m=750, lr=0.00445", linestyle='--', color='purple')


plt.axhline(y=0.5, color='red', linestyle='dotted', label=f"Threshold ({0.5 * 100}%)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.title("Accuracy for Fully-connected output, 1 fully-connected hidden layer vs Epoch")
plt.show()



'''
3b
'''
'''#model 1
model1 = model_convolutional(300, 5, 3)
optimizer1 = SGD(model1.parameters(), lr=.0089, momentum=0.95)

# We are using 20 epochs for this example. You may have to use more.
train_loss, train_accuracy1, val_loss, val_accuracy1 = train(
    model1, optimizer1, train_loader, val_loader, epochs=30)

test_loss, test_acc1 = evaluate(model1, test_loader)
print(f"Model 1's Test Accuracy: {test_acc1}, Validation Accuracy {max(val_accuracy1)}")

#model 2
model2 = model_convolutional(250, 3, 4)
optimizer2 = SGD(model2.parameters(), lr=0.0045000000000000005, momentum=0.95)

# We are using 20 epochs for this example. You may have to use more.
train_loss, train_accuracy2, val_loss, val_accuracy2 = train(
    model2, optimizer2, train_loader, val_loader, epochs=30)

test_loss, test_acc2 = evaluate(model2, test_loader)
print(f"Model 2's Test Accuracy: {test_acc2}, Validation Accuracy {max(val_accuracy2)}")

#model 3
model3 = model_convolutional(100, 3, 4)
optimizer3 = SGD(model3.parameters(), lr=0.0078000000000000005, momentum=0.93)

# We are using 20 epochs for this example. You may have to use more.
train_loss, train_accuracy3, val_loss, val_accuracy3 = train(
    model3, optimizer3, train_loader, val_loader, epochs=30)

test_loss, test_acc3 = evaluate(model3, test_loader)
print(f"Model 3's Test Accuracy: {test_acc3}, Validation Accuracy {max(val_accuracy3)}")

epochs = range(1, 31)
plt.plot(epochs, train_accuracy1, label="Model 1 Train Accuracy with M=300, k=5, N=3, lr=0.0089, momentum=0.95", linestyle='solid', color='blue')
plt.plot(epochs, val_accuracy1, label="Model 1 Validation Accuracy with M=300, k=5, N=3, lr=0.0089, momentum=0.95", linestyle='--', color='blue')

plt.plot(epochs, train_accuracy2, label="Model 2 Train Accuracy with M=250, k=3, N=4, lr=0.0045000000000000005, momentum=0.95", linestyle='solid', color='green')
plt.plot(epochs, val_accuracy2, label="Model 2 Validation Accuracy with M=250, k=3, N=4, lr=0.0045000000000000005, momentum=0.95", linestyle='--', color='green')

plt.plot(epochs, train_accuracy3, label="Model 3 Train Accuracy with M=100, k=3, N=4, lr=0.0078000000000000005, momentum=0.93", linestyle='solid', color='purple')
plt.plot(epochs, val_accuracy3, label="Model 3 Validation Accuracy with M=100, k=3, N=4, lr=0.0078000000000000005, momentum=0.93", linestyle='--', color='purple')


# Plotting the accuracy for all models
plt.axhline(y=0.65, color='red', linestyle='dotted', label=f"Threshold ({0.65 * 100}%)")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.title("Convolutional layer with max-pool and fully-connected output vs Epoch")
plt.show()
'''
