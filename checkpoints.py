import torch
import pathlib
from utils import Net
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import determined as det

from torchvision import datasets, transforms

def setup_models(device, hparams, core_context):
    model = Net(hparams)
    
    # try to load from checkpoint
    if core_context.info is None:
        info = det.get_cluster_info()
    else:
        info = core_context.info
    latest_checkpoint = info.latest_checkpoint
    trial_id = info.trial.trial_id
    
    if latest_checkpoint is None:
        epochs_completed = 0
        print('loaded model from scratch')
    else:
        with core_context.checkpoint.restore_path(latest_checkpoint) as path:
            checkpoint_directory = pathlib.Path(path)
            # load the model - depends on the framework
            with checkpoint_directory.joinpath("checkpoint.pt").open("rb") as f:
                model.load_state_dict(torch.load(f))
                print('loaded model from checkpoint')
                
            # read the training state - depends on the framework
            with checkpoint_directory.joinpath("state").open("r") as f:
                epochs_completed, ckpt_trial_id = [int(field) for field in f.read().split(",")]

        # if new trial, reset epochs
        if ckpt_trial_id != trial_id:
            print('new trial! restarting')
            epochs_completed = 0
        else:
            print(f'resuming at epoch {epochs_completed}')
                
    optimizer = optim.Adadelta(model.parameters(), lr=hparams["learning_rate"])
    scheduler = StepLR(optimizer, step_size=1, gamma=hparams["gamma"])
                    
    return model.to(device), optimizer, scheduler, epochs_completed


def checkpoint(core_context, model, epoch_idx, steps_completed):
    with core_context.checkpoint.store_path({"steps_completed": steps_completed}) as (path, storage_id):
        torch.save(model.state_dict(), path / "checkpoint.pt")
        with path.joinpath("state").open("w") as f:
            f.write(f"{epoch_idx+1},{core_context.info.trial.trial_id}")
            
            
def checkpoint_managed(core_context, model, epoch_idx, steps_completed, trial):
    with core_context.checkpoint.store_path({"steps_completed": steps_completed}) as (path, storage_id):
        torch.save(model.state_dict(), path / "checkpoint.pt")
        with path.joinpath("state").open("w") as f:
            f.write(f"{epoch_idx+1},{trial}")