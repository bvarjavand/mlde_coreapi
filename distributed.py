import logging

import torch.distributed as dist

import determined
import determined.core
from determined.experimental import core_v2 as core_context
import argparse
import uuid
import json
import torch
import torch.nn.functional as F
import utils, checkpoints


def train(model, device, train_loader, optimizer, scheduler, core_context, epoch_idx):
    size = dist.get_world_size()
    rank = dist.get_rank()
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (batch_idx + 1) % 100 == 0 and batch_idx % size == rank:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch_idx,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item()
            ), end='\r')
            
            core_context.train.report_training_metrics(
                steps_completed=(batch_idx + 1) + epoch_idx * len(train_loader), 
                metrics={"train_loss": loss.item(), "rank": rank}
            )

            
def main(hparams):
    device = torch.device("cuda")
    default, unmanaged = utils.core_setup()
    
    dist.init_process_group("gloo")
    distributed = core_context.DistributedContext.from_torch_distributed()
    size = dist.get_world_size()
    rank = dist.get_rank()

    with core_context.init_context(defaults=default, unmanaged=unmanaged, distributed=distributed) as context:
        # setup model from checkpoint, etc.
        train_loader, test_loader, model, optimizer, scheduler, epochs_completed = checkpoints.setup(device, hparams, context)
        
        # train while tracking metrics
        for epoch_idx in range(epochs_completed, hparams["epochs"]):
            steps_completed = (epoch_idx+1) * len(train_loader)
            train(model, device, train_loader, optimizer, scheduler, context, epoch_idx)
            
            if rank == 0:
                metrics.test( model, device,  test_loader, context, steps_completed=steps_completed)

                checkpoints.checkpoint(context, model, epoch_idx, steps_completed)
                    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hparams')
    args = parser.parse_args()

    main(json.loads(args.hparams))