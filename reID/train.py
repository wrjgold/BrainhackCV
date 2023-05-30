import numpy as np

import torch

from dataset import SiameseDataset
from model import SiameseNetwork
from transforms import Transforms
from inference import infer
# from utils import DeviceDataLoader, accuracy, get_default_device, to_device
import torch.nn as nn
import torch.nn.functional as F
# from test import predict_image
import torch as tt
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        #print(f'euclidean dist shape {euclidean_distance.shape}')
        loss_contrastive = torch.mean((label) * 0.5 * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * 0.5 * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        #print(f'cont loss shape {loss_contrastive.shape}')
        return loss_contrastive

def main():
    # Declare Siamese Network
    net = SiameseNetwork().cuda()
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = tt.optim.Adam(net.parameters(), lr=1e-2, weight_decay=0.005) #1e-3, 0.0005
    #load training data
    train_dir = '/content/drive/MyDrive/Brainhack/ReID/datasets/reducedDataset'
    train_dataset = SiameseDataset(train_dir, transform=Transforms())
    #print(train_dataset.shape)
    train_dataloader = DataLoader(train_dataset, num_workers=4,batch_size=32,shuffle=True)
    #train the model
    def train(epochs):
        loss=[] 
        counter=[]
        iteration_number = 0
        for epoch in range(1,epochs):
            train_loss = torch.Tensor([0]).cuda()
            for i, data in enumerate(train_dataloader,0):
                img0, img1 , label = data
                #print(img0.shape, img1.shape)
                #print(f'label is {label}')
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                optimizer.zero_grad()
                output1,output2 = net(img0,img1)
                #print(f'output1 is {output1}')
                loss_contrastive = criterion(output1,output2,label)
                #print(loss_contrastive)
                loss_contrastive.backward()
                optimizer.step()    
                '''
                for i in range (0,32):
                  print(torch.unsqueeze(img0[i], dim=0).shape)
                  pred = torch.Tensor([infer(net, torch.unsqueeze(img0[i], dim=0), torch.unsqueeze(img1[i], dim =0))])[0, 0]
                  print(pred)
                  train_loss = torch.cat((train_loss, pred)) #error with this line 
                  #TypeError: cat() received an invalid combination of arguments - got (Tensor, Tensor), but expected one of:
                  #* (tuple of Tensors tensors, int dim, *, Tensor out)
                  '''
                train_loss = train_loss + loss_contrastive
                #print (f'training loss {train_loss}')
            print("Epoch {}\n Current loss {}\n".format(epoch,train_loss.item()))
            iteration_number += 1
            counter.append(iteration_number)
            loss.append(loss_contrastive.item())
        plt.plot(counter, loss)
        return net
    #set the device to cuda
    device = torch.device('cuda' if tt.cuda.is_available() else 'cpu')
    print(f'device used {device}')
    model = train(25) #10 epochs
    torch.save(model.state_dict(), "/content/drive/MyDrive/Brainhack/reid_model.pt")
    print("Model Saved Successfully") 

main()