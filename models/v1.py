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