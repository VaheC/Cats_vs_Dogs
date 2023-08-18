import numpy as np

from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

class CatDogModel(object):

    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_loader = None
        self.val_loader = None

        self.writer = None

        self.total_epochs = 0

        self.losses = []
        self.val_losses = []

    def to(self, device):

        try:

            self.device = device

            self.model.to(self.device)

        except:

            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            print(f"{device} is not available. Using {self.device} instead!!!")

            self.model.to(self.device)

    def set_loaders(self, train_loader, val_loader=False):

        self.train_loader = train_loader

        self.val_loader = val_loader

    def set_tensorboard(self, name, folder="run"):

        self.writer = SummaryWriter(f"{folder}/{name}")

    def _create_train_step_fn(self):

        def get_train_loss(X, y):

            self.model.train()

            y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
        
        return get_train_loss
    
    def _create_val_step_fn(self):

        def get_val_loss(X, y):

            self.model.eval()

            y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)

            return loss.item()
        
        return get_val_loss
    
    def _get_minibatch_loss(self, validation=False):

        if validation:
            data_loader = self.val_loader
            step_fn = self._create_val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self._create_train_step_fn

        if data_loader is None:
            return None

        minibatch_losses = []

        for X_batch, y_batch in data_loader:

            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss = step_fn(X_batch, y_batch)

            minibatch_losses.append(loss)

        return np.mean(minibatch_losses)
    
    def set_seed(self, seed=42):

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)

        np.random.seed(seed)

    def train(self, n_epochs, seed=42):

        self.set_seed(seed)

        pbar = tqdm(range(n_epochs))

        for epoch in pbar:

            pbar.set_description(f"Epoch {epoch}")

            self.total_epochs += 1

            epoch_loss = self._get_minibatch_loss(validation=False)

            self.losses.append(epoch_loss)

            with torch.no_grad():

                epoch_val_loss = self._get_minibatch_loss(validation=True)

                self.val_losses.append(epoch_val_loss)

            if self.writer:
                scalars = {
                    'training': epoch_loss
                }
                if epoch_val_loss is not None:
                    scalars['validation'] = epoch_val_loss

                self.writer.add_scalars(
                    main_tag="loss",
                    tag_scalar_dict=scalars,
                    global_step=epoch
                )

        if self.writer:
            self.writer.flush()

    



