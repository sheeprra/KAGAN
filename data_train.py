from data_read import BrainDataset
import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import SimpleITK as sitk
import numpy as np

class Net(torch.nn.Module):
    def __init__(self，int N):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=N,out_channels=320,kernel_size=7,stride=2,padding=3)
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(in_channels=320,out_channels=640,kernel_size=3,stride=2,padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.conv3 = torch.nn.Conv2d(in_channels=640,out_channels=1280,kernel_size=3,stride=2,padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=8,stride=8)

        self.fc1 = torch.nn.Linear(1280, 2560)
        self.fc2 = torch.nn.Linear(2560, 640)
        self.fc3 = torch.nn.Linear(640, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

train_dir = '/data/Brain_data/train'
test_dir = '/data/Brain_data/test'
new_size = (160, 256, 256)
train_dataset = BrainDataset(root_dir=train_dir,new_size=new_size)
test_dataset = BrainDataset(root_dir=test_dir,new_size=new_size)
print(len(train_dataset),len(test_dataset))

print(train_dataset[0][0].shape)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)
print(len(train_dataloader),len(test_dataloader))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


model = Net().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10
total_step = len(train_dataloader)
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_dataloader):

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))


