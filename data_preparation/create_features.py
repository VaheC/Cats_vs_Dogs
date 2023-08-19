import os

from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.image as img

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import Compose, ToTensor, CenterCrop


def get_features(model, data_loader, device=None, is_test=False):

    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    features = None

    if is_test:
        pass
    else:
        targets = None
        folds = None

    for i, data in tqdm(enumerate(data_loader)):

        if is_test:
            X = data
        else:
            X, y, fold = data

        X = X.to(device)

        model.eval()

        if i == 0:

            features = model(X).detach().cpu().numpy()

            if is_test:
                pass
            else:
                targets = y.cpu().numpy()
                folds = fold

        else:

            features = np.concatenate(
                [features, model(X).detach().cpu().numpy()])

            if is_test:
                pass
            else:
                targets = np.concatenate([targets, y.cpu().numpy()])
                folds = np.concatenate([folds, fold])

    if is_test:
        return features
    else:
        return features, targets, folds


class CustomDataset(Dataset):

    def __init__(self, df, transform=None, is_test=False):

        self.transform = transform

        self.is_test = is_test

        if self.is_test:
            self.df = df[['file']]
        else:
            self.df = df[['file', 'target', 'fold', 'pet']]

    def __getitem__(self, index):

        if not self.is_test:
            image_path = self.df.iloc[index]['pet']
            image_path = image_path + f"/{self.df.iloc[index]['file']}"
        else:
            image_path = self.df.iloc[index]['file']

        if self.is_test:
            image_path = os.path.join('test', image_path)
        else:
            image_path = os.path.join('train', image_path)

        self.X = img.imread(image_path)
        # h_start = int(self.X.shape[0] / 2)
        # w_start = int(self.X.shape[1] / 2)
        # self.X = self.X[h_start:, w_start:, :]
        # self.X = ToTensor()(self.X)
        # self.X = self.X[:, :240, :240]

        if self.transform is not None:
            self.X = self.transform(self.X)
        else:
            self.X = ToTensor()(self.X)

        if self.is_test:

            return self.X

        else:

            self.y = self.df.iloc[index]['target']
            self.y = torch.as_tensor(self.y)

            self.fold = self.df.iloc[index]['fold']

            return (self.X, self.y, self.fold)

    def __len__(self):
        return len(self.df)


def create_train_data(model, data_df):
    train_composer = Compose(
        [
            ToTensor(),
            CenterCrop(224)

        ]
    )

    train_data = CustomDataset(df=data_df, transform=train_composer)

    train_loader = DataLoader(
        dataset=train_data,
        shuffle=False,
        batch_size=16
    )

    X_train, y_train, folds = get_features(
        model=model, data_loader=train_loader, device='cuda')

    resnet18_df = pd.DataFrame(X_train)
    resnet18_df.columns = [f"feat{i}" for i in range(1, X_train.shape[1]+1)]
    resnet18_df['target'] = y_train
    resnet18_df['fold'] = folds
    resnet18_df.to_csv('train_resnet18.csv', index=False)


def create_test_data(model, data_df):

    test_composer = Compose(
        [
            ToTensor(),
            CenterCrop(224)

        ]
    )

    test_data = CustomDataset(
        df=data_df, transform=test_composer, is_test=True)

    test_loader = DataLoader(
        dataset=test_data,
        shuffle=False,
        batch_size=16
    )

    X_test = get_features(
        model=model, data_loader=test_loader, device='cuda', is_test=True)

    resnet18_df = pd.DataFrame(X_test)
    resnet18_df.columns = [f"feat{i}" for i in range(1, X_test.shape[1]+1)]
    resnet18_df.to_csv('test_resnet18.csv', index=False)
