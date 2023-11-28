import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
from determined.experimental import core_v2 as core_context


class Net(nn.Module):
    def __init__(self, hparams, **kwargs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, hparams["n_filters1"], 3, 1)
        self.conv2 = nn.Conv2d(hparams["n_filters1"], hparams["n_filters2"], 3, 1)
        self.dropout1 = nn.Dropout(hparams["dropout1"])
        self.dropout2 = nn.Dropout(hparams["dropout2"])
        self.fc1 = nn.Linear(144 * hparams["n_filters2"], 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    
def core_setup():
    external_experiment_id = str(uuid.uuid4())
    external_trial_id = str(uuid.uuid4())

    default_context = core_context.DefaultConfig(
        name = 'detached', checkpoint_storage='/determined_shared_fs')

    unmanaged_context = core_context.UnmanagedConfig(
        workspace = 'Bijan', project='test',
        external_experiment_id = external_experiment_id,
        external_trial_id = external_trial_id)
    
    return default_context, unmanaged_context