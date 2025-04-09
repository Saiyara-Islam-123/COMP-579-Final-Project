from torch import optim

from CNN import *
import pandas as pd
import torch
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset():
    y = pd.read_csv("out.csv")["y"]
    y_transformed = []
    for i in range(len(y)):
        if y.iloc[i] == '-':
            y_transformed.append(10)
        else:
            y_transformed.append(int(y.iloc[i]))


    X = []
    for i in range(0, len(os.listdir("chars data"))):
        img = cv2.imread("chars data/" + str(i) + ".png", cv2.IMREAD_GRAYSCALE)
        X.append(img)

    y = np.array(y_transformed)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=43)


    X_train = torch.tensor(np.array(X_train))
    X_test = torch.tensor(np.array(X_test))
    y_train = torch.tensor(np.array(y_train))
    y_test = torch.tensor(np.array(y_test))
    return X_train.to(torch.float32), X_test.to(torch.float32), y_train.to(torch.float32), y_test.to(torch.float32)


X_train, X_test, y_train, y_test = get_dataset()

model = CNN(input_size=108)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

batches = list(torch.split(X_train, 32))
batches_of_labels = list(torch.split(y_train, 32))

for epoch in range(10):
    for i in range(len(batches)):
        optimizer.zero_grad()
        batch = batches[i]
        outputs = model(batch.reshape( batch.shape[0], 1, batch.shape[1], batch.shape[2]))
        loss = loss_fn(outputs, batches_of_labels[i].long())
        loss.backward()
        optimizer.step()
    print(epoch, loss.item())

def acc(model, test_inputs, test_labels):
    pred_labels = model(test_inputs).argmax(dim=1)

    acc = (pred_labels == test_labels).float().mean().item()

    print("\nAccuracy = ")
    print(acc * 100)
    return acc

#torch.save(model, "cnn_trained.pth")
acc(model, X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]), y_test.reshape(y_test.shape[0], 1))


