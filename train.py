from torch.utils.data import DataLoader, Dataset
import os
import cv2
from torchvision import transforms
from unet_pl import *
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

transforms = transforms.Compose([transforms.ToTensor(),
                                 #transforms.Normalize()
                                 ])

class CrackDataset(Dataset):
    def __init__(self, img_path, lab_path, img_size = (360, 640), is_transform = True):
        self.img_path = img_path
        self.img_files = os.listdir(img_path)

        self.lab_path = lab_path
        self.lab_files = os.listdir(lab_path)

        self.transforms = transforms
        self.len = len(self.img_files)
        self.img_size = img_size
        self.n_classes = 2
        self.is_transform = is_transform

        self.images = [os.path.join(img_path, x) for x in self.img_files]
        self.labels = [os.path.join(lab_path, x) for x in self.lab_files]


    def __len__(self):
        return(self.len)

    def __getitem__(self, index):
        x_data = Image.open(self.images[index])
        y_data = Image.open(self.labels[index])

        x = self.transforms(x_data)
        y = np.array(y_data)
        y = torch.from_numpy(y)# modify the channel to 3

        return x, y[:, :, 0]

    def get_name(self, i):
        return(self.images[i])

train_dataset = CrackDataset('iamge_path',
                             'label_path')



val_dataset = CrackDataset('iamge_path',
                           'label_path')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle = True)


model = unet_pl()

device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
model.to(device)

epoch = 25
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)


if __name__ == '__main__':
    for epoch in tqdm(range(epoch)):
        model.train()
        print('start')
        for i, data in enumerate(train_loader):
            x, y = data
            x, y = x.to(device), y.to(device, dtype = torch.int64) #the dtype of y must be int/long
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        print(loss / len(train_loader))
        print('epoch end')
        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(xv.to(device)), yv.to(device, dtype = torch.int64)) for xv, yv in val_loader)
            print('%d  %s' % (epoch, val_loss / len(val_loader)))
            print('val end')
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), '%d.pth' % epoch)









