from torch import optim

from CNN import *
import torch
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset():
    y = []

    X = []
    for i in range(0, len(os.listdir("Aug Data"))):
        file_name = os.listdir("Aug Data")[i]
        label = file_name.split(" ")[0].strip(".png")
        img = cv2.imread("Aug Data/" + file_name, cv2.IMREAD_GRAYSCALE)
        X.append(img)
        y.append(int(label))
        print(i)

    y = np.array(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)


    X_train = torch.tensor(np.array(X_train))
    X_test = torch.tensor(np.array(X_test))
    y_train = torch.tensor(np.array(y_train))
    y_test = torch.tensor(np.array(y_test))
    return X_train.to(torch.float32), X_test.to(torch.float32), y_train.to(torch.long), y_test.to(torch.long)


X_train, X_test, y_train, y_test = get_dataset()



model = CNN(input_size=108)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)

batches = list(torch.split(X_train, 30))
batches_of_labels = list(torch.split(y_train, 30))

print(X_train.shape)

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
    model.eval()
    pred_labels = model(test_inputs).argmax(dim=1).numpy()

    acc = (pred_labels == test_labels).float().mean().item()

    print("\nAccuracy = ")
    print(acc * 100)
    return acc

#torch.save(model, "cnn_trained.pth")
acc(model, X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]), y_test.reshape(y_test.shape[0], 1).long())


