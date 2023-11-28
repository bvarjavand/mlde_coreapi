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
import utils, base, checkpoints, metrics


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
    
    dist.init_process_group("gloo")
    distributed = core_context.DistributedContext.from_torch_distributed()
    size = dist.get_world_size()
    rank = dist.get_rank()

    with det.core.init(distributed=distributed) as context:
        # setup model from checkpoint, etc.
        train_loader, test_loader  = base.setup_datasets(device, hparams)
        model, optimizer, scheduler, epochs_completed = checkpoints.setup_models(device, hparams, context)
        epoch_idx = epochs_completed
        last_checkpoint_batch = None
                    
                    
        for op in context.searcher.operations():
            # NEW: Use a while loop for easier accounting of absolute lengths.
            while epoch_idx < op.length:
                steps_completed = (epoch_idx+1) * len(train_loader)
                train(model, device, train_loader, optimizer, scheduler, context, epoch_idx)
                epoch_idx += 1
                
                if rank == 0:
                    metrics.test( model, device,  test_loader, context, steps_completed=steps_completed)

                    checkpoints.checkpoint_managed(context, model, epoch_idx, steps_completed, info.trial.trial_id)
                
                if context.distributed.rank == 0:
                    op.report_progress(epoch_idx)
                
                if context.preempt.should_preempt():
                    return

            # NEW: After training for each op, validate and report the
            # searcher metric to the master.
            op.report_completed(test_loss)

if __name__ == '__main__':
    import determined as det
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    print(hparams)
    main(hparams)