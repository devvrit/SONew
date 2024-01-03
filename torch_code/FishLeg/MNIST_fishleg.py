import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import time
import os
import sys
import matplotlib.pyplot as plt
import torch.optim as optim
from datetime import datetime
is_nni = False
if is_nni:
    import nni

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import default_collate

from data_utils import read_data_sets

torch.set_default_dtype(torch.float32)

from FishLeg import FishLeg, FISH_LIKELIHOODS, initialise_FishModel

seed = 13
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = read_data_sets("MNIST", "../data/", if_autoencoder=True)

## Dataset
train_dataset = dataset.train
test_dataset = dataset.train

batch_size = 1000

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

aux_loader = torch.utils.data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
)

model = nn.Sequential(
    nn.Linear(784, 1000, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(1000, 500, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(500, 250, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(250, 30, dtype=torch.float32),
    nn.Linear(30, 250, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(250, 500, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(500, 1000, dtype=torch.float32),
    nn.ReLU(),
    nn.Linear(1000, 784, dtype=torch.float32),
)

########## Default params start #############
########## Default params start #############
eta_adam = 1e-4

lr = 0.0018709607159126847
beta = 1-0.0913584816351351

aux_lr = 0.00005543844571757454
aux_eps = 8.995872632759885e-9
damping = 0.004110608813872052
scale_factor=1
########## Default params end ##############
########## Default params end ##############


############ NNI Start ################
############ NNI Start ################
if is_nni:
    optimized_params = nni.get_next_parameter()
    beta = 1.0 - optimized_params['momentum']
    lr = optimized_params['base_lr']
    aux_lr = optimized_params['aux_lr']
    damping = optimized_params['damping']
    aux_eps = optimized_params['eps']
############ NNI Ends ################
############ NNI Ends ################


weight_decay = 0.0
update_aux_every = 10

initialization = "normal"
normalization = True

model = initialise_FishModel(
    model, module_names="__ALL__", fish_scale=scale_factor / damping
)

model = model.to(device)

likelihood = FISH_LIKELIHOODS["bernoulli"](device=device)

writer = SummaryWriter(
    log_dir=f"runs/MNIST_fishleg/lr={lr}_auxlr={aux_lr}/{datetime.now().strftime('%Y%m%d-%H%M%S')}",
)

opt = FishLeg(
    model,
    aux_loader,
    likelihood,
    lr=lr,
    beta=beta,
    weight_decay=weight_decay,
    aux_lr=aux_lr,
    aux_betas=(0.9, 0.999),
    aux_eps=aux_eps,
    damping=damping,
    update_aux_every=update_aux_every,
    writer=writer,
    method="antithetic",
    method_kwargs={"eps": 1e-4},
    precondition_aux=True,
)

epochs = 100
num_steps_per_epoch = len(train_dataset)//batch_size

# Learning Rate Schedule
warmup_epochs = 5
lr_vec = np.concatenate([np.linspace(0, lr, warmup_epochs),
    np.linspace(lr, 0, epochs-warmup_epochs+2)[1:-1]], axis=0)

st = time.time()
eval_time = 0

for epoch in range(1, epochs + 1):
    with tqdm(train_loader, unit="batch") as tepoch:
        running_loss = 0
        for g in opt.param_groups:
            g['lr'] = lr_vec[epoch]
        for n, (batch_data, batch_labels) in enumerate(tepoch, start=1):

            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            opt.zero_grad()
            output = model(batch_data)

            loss = likelihood(output, batch_labels)
            tepoch.set_description(f"Epoch {epoch} step {n}, loss: {loss}")

            running_loss += loss.item()

            loss.backward()
            opt.step()
            tepoch.set_postfix(loss=loss.item())

        eval_time = 0
        if epoch%20 == 0:
            et = time.time()
            model.eval()
            running_test_loss = 0
            for m, (test_batch_data, test_batch_labels) in enumerate(test_loader):
                test_batch_data, test_batch_labels = test_batch_data.to(
                    device
                ), test_batch_labels.to(device)
                test_output = model(test_batch_data)
                test_loss = likelihood(test_output, test_batch_labels)
                running_test_loss += test_loss.item()
            running_test_loss /= m
            print("running_test_loss:", running_test_loss)
            if is_nni:
                nni.report_intermediate_result({ 'default':running_test_loss})
                if epoch == epochs:
                    nni.report_final_result({'default':running_test_loss})
            eval_time += time.time() - et

        model.train()
        epoch_time = time.time() - st - eval_time
        writer.add_scalar("Loss/train", running_loss / n, epoch)

        # Write out the losses per wall clock time
        writer.add_scalar("Loss/train/time", running_loss / n, epoch_time)
