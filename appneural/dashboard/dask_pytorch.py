import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataset import random_split

import numpy as np
import matplotlib.pyplot as plt

import dask.dataframe as pd


TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"


torch.manual_seed(9)
torch.cuda.manual_seed(9)
torch.cuda.manual_seed_all(9)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def build_trainval_loaders(datafield, batch_size=64, val_size=0.1):

    data_n = np.array(datafield)
    data_x_t = torch.tensor(data_n[:,1:]).to(device).float().reshape(-1, 1, 28, 28)
    data_x_t = data_x_t.float() / 255 # normalize
    data_y_t = torch.tensor(data_n[:,0]).to(device)

    dataset = torch.utils.data.TensorDataset(data_x_t, data_y_t)

    # split for validation set
    val_ds_size = int(len(dataset) * val_size)
    sizes = [len(dataset) - val_ds_size, val_ds_size]
    train_dataset, val_dataset = random_split(dataset, sizes)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader

def build_test_loader(datafield, batch_size=64):

    data_n = np.array(datafield)
    data_x_t = torch.tensor(data_n).to(device).float().reshape(-1, 1, 28, 28)
    data_x_t = data_x_t.float() / 255 # normalize

    dataset = torch.utils.data.TensorDataset(data_x_t)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

class CNNNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv0 = nn.Conv2d(1, 32, (3, 3), bias=False)
        self.norm0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(32, 32, (3, 3), bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, (3, 3), bias=False)
        self.norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 32, (5, 5), bias=False, stride=2, padding=2)
        self.norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.4)

        self.conv4 = nn.Conv2d(32, 64, (3, 3), bias=False)
        self.norm4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(64, 64, (3, 3), bias=False)
        self.norm5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(64, 64, (5, 5), bias=False, stride=2, padding=2)
        self.norm6 = nn.BatchNorm2d(64)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.4)

        self.flatten = nn.Flatten(1, -1)

        self.fc7 = nn.Linear(1024, 128)
        self.norm7 = nn.BatchNorm1d(128)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(0.4)

        self.fc8 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.relu0(self.norm0(self.conv0(x)))
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.relu3(self.norm3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.relu4(self.norm4(self.conv4(x)))
        x = self.relu5(self.norm5(self.conv5(x)))
        x = self.relu6(self.norm6(self.conv6(x)))
        x = self.dropout6(x)
        x = self.flatten(x)
        x = self.relu7(self.norm7(self.fc7(x)))
        x = self.dropout7(x)

        return self.fc8(x)

def train(model, loss_fn, optimizer, data_loader):

    model.train()

    losses = 0
    losses_cnt = 0
    correct_cnt = 0
    total_cnt = 0

    for x, y in data_loader:

        out = model(x)

        correct_cnt += int(sum(y == torch.argmax(out, dim=1)))

        total_cnt += len(x)

        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach the loss or graph doesn't get freed and memory keeps
        # increasing
        losses += loss.detach().item()
        losses_cnt += 1



    return losses / losses_cnt, correct_cnt / total_cnt

def evaluate(model, loss_fn, data_loader):

    model.eval()

    # validate

    losses = 0
    losses_cnt = 0
    correct_cnt = 0
    total_cnt = 0

    with torch.no_grad():

        for x, y in data_loader:

            out = model(x)

            loss = loss_fn(out, y)

            correct_cnt += int(sum(y == torch.argmax(out, dim=1)))
            total_cnt += len(x)

            # detach the loss or graph doesn't get freed and memory keeps
            # increasing
            losses += loss.detach().item()
            losses_cnt += 1

    return losses / losses_cnt, correct_cnt / total_cnt


def main():

    n_epochs = 20
    batch_size = 16

    print("Prepare data")
    train_loader, val_loader = build_trainval_loaders(
            pd.read_csv(TRAIN_CSV), batch_size=batch_size, val_size=0.2)

    print("Setup model")
    model = CNNNet()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=8, gamma=0.1)

    losses = []
    accs   = []
    losses_val = []
    accs_val = []

    print(f"Train: epochs={n_epochs}")
    for epoch in range(1, n_epochs+1):

        loss, acc = train(model, loss_fn, optimizer, train_loader)

        loss_val, acc_val = evaluate(model, loss_fn, val_loader)

        scheduler.step()

        losses.append(loss)
        accs.append(acc)
        losses_val.append(loss_val)
        accs_val.append(acc_val)

        print (f"Epoch {epoch:3d} | Train loss {loss:.6f} acc {acc:.4f} | "
                f"Validation loss {loss_val:.6f} acc {acc_val:.4}")

    print("Plot")
    epochs = range(1, n_epochs+1)
    fig, axis = plt.subplots(2, figsize=(10, 10))
    fig.tight_layout(h_pad=5)

    axis[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    axis[0].plot(epochs, losses, "b", label="loss_train")
    axis[0].plot(epochs, losses_val, "g", label="loss_validate")
    axis[0].legend()

    axis[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axis[1].plot(epochs, accs, "b", label="accuracy_train")
    axis[1].plot(epochs, accs_val, "g", label="accuracy_validate")
    axis[1].legend()
# TODO save model  should have

    plt.show()


# to run :
if __name__ == "__main__":
    main()

# Prepare data
# Setup model
# Train: epochs=20
# Epoch   1 | Train loss 0.232913 acc 0.9351 | Validation loss 0.049689 acc 0.9843
# -------------------
# Epoch  20 | Train loss 0.017774 acc 0.9948 | Validation loss 0.017100 acc 0.9944
# Plot