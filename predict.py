import torch
from torchvision import transforms
from ptsemseg.models.unet_pl import unet_pl
import os
from train_pl import CrackDataset
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import torch.nn.functional as F
import logging
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


logging.getLogger().setLevel(logging.INFO)
tensor2img = transforms.ToPILImage()
colors = np.array([[0, 0, 0],
                   [255, 255, 255]])

test_images = 'G:\paper\pytorch-semseg-master\data/val_imgs'
test_labels = 'G:\paper\pytorch-semseg-master\data/val_labs'
weights_path = 'G:\paper\pytorch-semseg-master/24.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = unet_pl()


model.load_state_dict(torch.load(weights_path))
model.eval()
model.to(device)



testdataset = CrackDataset(test_images,
                           test_labels)

test_loader = DataLoader(testdataset, batch_size = 2, shuffle = False)

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device, dtype = torch.float32), labels.to(device)
        logging.info('processing')
        y_pred = model(images)
        y_prob = torch.softmax(y_pred, dim = 1)


        out = torch.argmax(y_prob, dim = 1)

        masks = [colors[p.cpu()] for p in out]

        for mask in masks:
            print(mask.shape)
            #mask = mask.permute()
            plt.matshow(mask, cmap = 'gray')
            plt.show()

        cv2.waitKey()


        #print(np.sum(out))


'''

torch.save(model, '\model.pkl')

model = torch.load('\model.pkl')

'''
