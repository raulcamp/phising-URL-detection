import pickle
import random
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch import nn, Tensor, optim
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def encode(url):
    encoding = ""
    for c in url:
        encoding += str(ord(c))
    return int(encoding)

def get_dataloaders(df, batch_size, generator, train_size, validation_size, test_size):
    dataset = torch.utils.data.TensorDataset(torch.tensor(
        df['URL'].values), torch.tensor(df['Label'].values, dtype=torch.float32))

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size, test_size], generator=generator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, worker_init_fn=seed_worker, generator=generator)
    return (train_dataloader, validation_dataloader, test_dataloader)


class LSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_layers,
                 dropout):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)
        self.linear = nn.Linear(2*hidden_size, 1)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x, _ = self.cell(x)
        # for LTSMVis
        outputs = x.clone()
        h_n = _[0]
        x = self.linear(x[:, -1, :])
        return x, outputs, h_n


def train(model, train_loader, validation_loader, n_epochs, optimizer, device):
    loss_fn = nn.BCEWithLogitsLoss()
    training_losses, validation_losses = [], []
    for _ in tqdm(range(n_epochs)):
        epoch_loss = 0
        model.train()

        for url, label in train_loader:
            # print(url.shape)
            preds, _, _ = model(url.to(device))  # forward pass
            # to avoid complaints about different devices
            label = label.to(device)

            loss = loss_fn(preds[:, 0], label)  # computing the loss
            optimizer.zero_grad()  # zeroing the gradients of the model parameters
            loss.backward()  # backward pass
            optimizer.step()  # model parameters

            epoch_loss += loss.item()
        print('\t Epoch Train Loss: ', epoch_loss / len(train_loader))
        training_losses.append(epoch_loss/len(train_loader))

        epoch_loss = 0
        model.eval()
        for url, label in validation_loader:
            preds, _, _ = model(url.to(device))  # forward pass
            # to avoid complaints about different devices
            label = label.to(device)

            loss = loss_fn(preds[:, 0], label)  # computing the loss
            epoch_loss += loss.item()
        print('\t Epoch valid Loss: ', epoch_loss / len(validation_loader))
        validation_losses.append(epoch_loss/len(validation_loader))

    return model, training_losses, validation_losses


def test(model, test_dataloader, device):
    # We use BinaryCrossEntropyLoss for our logistic regression task
    loss_fn = nn.BCEWithLogitsLoss()

    urls, labels = next(iter(test_dataloader))[0], next(iter(test_dataloader))[1]
    # Our predictions
    outputs = model(urls.to(device))[0]

    predictions = torch.round(torch.sigmoid(outputs))

    # Losses
    loss = loss_fn(outputs[:, 0], labels.to(device))  # computing the loss

    # We need numpy arrays for metrics
    predictions, labels = predictions.cpu().detach().numpy(), labels.cpu().detach().numpy()

    return (metrics.accuracy_score(predictions, labels, normalize=True),
            metrics.precision_score(predictions, labels),
            metrics.recall_score(predictions, labels),
            metrics.f1_score(predictions, labels),
            loss/len(test_dataloader))


def train_model(seed=0,
                device='cuda',
                batch_size=64,
                learning_rate=1e-3,
                train_size=8000,
                validation_size=500,
                test_size=1500,
                n_epochs=10,
                **kwargs):

    # We ensure our results are reproducible
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Reproducibility for GPUs
    if device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # We get our dataset based on our language
    df = pd.read_csv('script/data/Phishing_Dataset.csv')
    print(len(df))
    data = get_dataloaders(df, batch_size, generator, train_size, validation_size, test_size)

    # We construct our model
    model = LSTM(vocab_size=1, **kwargs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # We train and test our model
    model, train_losses, validation_losses = train(model, data[0], data[1], n_epochs, optimizer, device)
    accuracy, precision, recall, f1, test_loss = test(model, data[2], device)

    # We save our model
    pickle.dump(model, open("script/lstm_model", "wb"))

    return model, (train_losses, validation_losses), (accuracy, precision, recall, f1), test_loss


if __name__ == '__main__':
    # train
    model_parameters = {
        'embedding_dim': 200,
        'hidden_size': 200,
        'num_layers': 3,
        'dropout': 0.2
    }
    train_model(
        seed=0,
        batch_size=32,
        train_size=3500,
        validation_size=500,
        test_size=500,
        n_epochs=5,
        **model_parameters)
