from torch.utils.tensorboard import SummaryWriter
import torch

from torchvision.transforms import ToTensor, CenterCrop

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.image as img
plt.style.use('fivethirtyeight')


class CatDogModel(object):

    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.writer = None

        self.total_epochs = 0

        self.losses = []
        self.val_losses = []

        self.train_step_fn = self._create_train_step_fn()
        self.val_step_fn = self._create_val_step_fn()

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

            y_hat = self.model(X).view(-1, )

            loss = self.loss_fn(y_hat, y)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()

        return get_train_loss

    def _create_val_step_fn(self):

        def get_val_loss(X, y):

            self.model.eval()

            y_hat = self.model(X).view(-1, )

            loss = self.loss_fn(y_hat, y)

            return loss.item()

        return get_val_loss

    def _get_minibatch_loss(self, validation=False):

        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

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

    def save_checkpoint(self, filename):

        checkpoint = {
            "loss": self.losses,
            "val_loss": self.val_losses,
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):

        checkpoint = torch.load(filename)

        self.model.load_state_dict(
            checkpoint['model_state_dict']
        )

        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )

        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.model.train()

    def predict(self, x):

        self.model.eval()

        x_tensor = torch.as_tensor(x).float()

        y_hat_tensor = self.model(x_tensor.to(self.device))

        self.model.train()

        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):

        fig = plt.figure(figsize=(8, 4))
        plt.plot(self.losses, label='training', c='b')

        if self.val_loader:
            plt.plot(self.val_losses, label='validation', c='r')

        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()

        return fig
    
    @staticmethod
    def plot_misclassified_images(y, y_hat, image_names, bce_loss, n_images=None, pet_dict = {1: "cat", 0: "dog"}):

        misclassified_images = [
            (f"train/{pet_dict[y[i]]}/{image_names[i]}", bce_loss[i])
            for i in range(len(image_names)) 
            if y[i]!=y_hat[i]
        ]

        misclassified_images = sorted(misclassified_images, key=lambda x: x[1], reverse=True)

        if n_images is not None:
            misclassified_images = misclassified_images[:n_images]

        n_images_per_row = 10

        temp_remainder = len(misclassified_images) % n_images_per_row

        n_rows = int((len(misclassified_images) - temp_remainder) / n_images_per_row)

        if temp_remainder != 0:
            n_rows += 1

        fig_width = 2 * n_images_per_row
        fig_height = 2 * n_rows

        fig = plt.figure(figsize=(fig_width, fig_height))

        for i in range(len(misclassified_images)):

            plt.subplot(n_rows, n_images_per_row, i+1)
            image_mat = img.imread(misclassified_images[i][0])
            image_tensor = ToTensor()(image_mat.copy())
            image_tensor = CenterCrop(224)(image_tensor)
            new_image_mat = image_tensor.cpu().numpy()
            new_image_mat = np.transpose(new_image_mat, (1, 2, 0))
            plt.imshow(new_image_mat)
            plt.grid(False)
            plt.axis('off')

        plt.tight_layout()