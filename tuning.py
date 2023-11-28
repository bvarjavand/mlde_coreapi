import metrics

def train(model, device, train_loader, optimizer, scheduler, core_context, epoch_idx, op):
    
    
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
            
            core_context.train.report_training_metrics(
                steps_completed=(batch_idx + 1) + epoch_idx * len(train_loader), 
                metrics={"train_loss": loss.item()}
            )


def test(model, device, test_loader, core_context, steps_completed, op):
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

    core_context.train.report_validation_metrics(
        steps_completed=steps_completed, 
        metrics={"test_loss": test_loss})