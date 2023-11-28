import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from torchvision import datasets, transforms
from utils import Net

def setup_datasets(device, hparams):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **{"batch_size": hparams["batch_size"]})
    test_loader = torch.utils.data.DataLoader(test_dataset, **{"batch_size": hparams["test_batch_size"]})
    
    return train_loader, test_loader


def setup_models(device, hparams):
    model = Net(hparams)
    
    optimizer = optim.Adadelta(model.parameters(), lr=hparams["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=hparams["gamma"])
    
    return model.to(device), optimizer, scheduler


def train(model, device, train_loader, optimizer, scheduler, epoch_idx):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (batch_idx + 1) % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch_idx,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item()
            ), end='\r')


def test(model, device, test_loader):
    model.eval()
    test_loss = correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
    ), end='\r')