import torch
from CNN import *
import cv2
import numpy as np



model = torch.load('cnn_trained.pth')
model.eval()

img = torch.tensor(np.array(cv2.imread("char3.png", cv2.IMREAD_GRAYSCALE)).reshape(1, 108, 108)).float()
print(model(img).argmax(dim=1).item())