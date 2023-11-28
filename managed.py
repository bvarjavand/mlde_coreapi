import torch
import determined as det
import base, metrics, checkpoints

def main(core_context):
    torch.manual_seed(1)
    info = det.get_cluster_info()
    hparams = info.trial.hparams
    print(hparams)
        
    train_kwargs = {"batch_size": hparams["batch_size"]}
    test_kwargs = {"batch_size": hparams["test_batch_size"]}
    if torch.cuda.is_available():
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    train_loader, test_loader  = base.setup_datasets(device, hparams)
    model, optimizer, scheduler, epochs_completed = checkpoints.setup_models(device, hparams, core_context) # loading models from checkpoint
    
    for epoch_idx in range(epochs_completed, hparams["epochs"]):
        steps_completed = epoch_idx * len(train_loader)
        metrics.train(model, device, train_loader, optimizer, scheduler, core_context, epoch_idx)
        metrics.test( model, device,  test_loader, core_context, steps_completed=steps_completed)
        checkpoints.checkpoint_managed(core_context, model, epoch_idx, steps_completed, info.trial.trial_id)


if __name__ == "__main__":
    with det.core.init() as core_context:
        main(core_context=core_context)