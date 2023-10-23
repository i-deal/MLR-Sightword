from mVAE import vae, VAEshapelabels, VAEcolorlabels, dataset_builder
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset, ConcatDataset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy


vae_shape_labels= VAEshapelabels(xlabel_dim=20, hlabel_dim=7,  zlabel_dim=4)

vae_color_labels= VAEcolorlabels(xlabel_dim=10, hlabel_dim=7,  zlabel_dim=4)

if torch.cuda.is_available():
    vae.cuda()
    vae_shape_labels.cuda()
    vae_color_labels.cuda()
    print('CUDA')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    vae.load_state_dict(checkpoint['state_dict'])

    for parameter in vae.parameters():
        parameter.requires_grad = False
    vae.eval()
    return vae


def load_checkpoint_shapelabels(filepath):
    checkpoint = torch.load(filepath)
    vae_shape_labels.load_state_dict(checkpoint['state_dict_shape_labels'])
    for parameter in vae_shape_labels.parameters():
        parameter.requires_grad = False
    vae_shape_labels.eval()
    return vae_shape_labels

def load_checkpoint_colorlabels(filepath):
    checkpoint = torch.load(filepath)
    vae_color_labels.load_state_dict(checkpoint['state_dict_color_labels'])
    for parameter in vae_color_labels.parameters():
        parameter.requires_grad = False
    vae_color_labels.eval()
    return vae_color_labels

optimizer = optim.Adam(vae.parameters())

optimizer_shapelabels= optim.Adam(vae_shape_labels.parameters())
optimizer_colorlabels= optim.Adam(vae_color_labels.parameters())

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')

    #Loss=nn.CrossEntropyLoss()(z_label,Variable(labels))

    #z_label=torch.tensor(z_label,requires_grad=True)
    #e=nn.CrossEntropyLoss()(z_label,z_shape)
    e=criterion(label_act,image_act)
   # print('the error is')
    #print(e)
    return e

def train_labels(epoch):
    global colorlabels, numcolors
    #colorlabels = np.random.randint(0,2, 100000)
    #print(colorlabels)
    #colors= torch.tensor([0,1])
    # colors=colors.clone().detach()

   # colorlabels = torch.cat(100000* [colors])
    

    numcolors = 0
    train_loss_shapelabel = 0
    train_loss_colorlabel = 0

    vae_shape_labels.train()
    vae_color_labels.train()

    dataiter = iter(train_loader)
    red_labels=torch.tensor([0,1,2,3,4,5,6,7,8,9]) #only ten colors

    # labels_color=0

    for i in tqdm(range(len(train_loader))):
        optimizer_shapelabels.zero_grad()
        optimizer_colorlabels.zero_grad()


        image, labels = dataiter.next()
       
        
        labels_forcolor=labels.clone()
        for col in red_labels:
            labels_forcolor[labels_forcolor==10+col]=col
            
        
        image = image.cuda()
        labels = labels.cuda()
        input_oneHot = F.one_hot(labels, num_classes=20) # 20 classes for f-mnist
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        labels_color = labels_forcolor  # get the color labels
        
        labels_color = labels_color.cuda()
        
        #print(labels_color)
        
        color_oneHot = F.one_hot(labels_color, num_classes=10)
        color_oneHot = color_oneHot.float()
        color_oneHot = color_oneHot.cuda()



        z_shape_label = vae_shape_labels(input_oneHot)
        z_color_label = vae_color_labels(color_oneHot)

        z_shape, z_color = image_activations(image)


        loss_of_shapelabels = loss_label(z_shape_label, z_shape)
        loss_of_shapelabels.backward()
        train_loss_shapelabel += loss_of_shapelabels.item()
            # print(train_loss_label)
        optimizer_shapelabels.step()

        loss_of_colorlabels = loss_label(z_color_label, z_color)
        loss_of_colorlabels.backward()
        train_loss_colorlabel += loss_of_colorlabels.item()
            # print(train_loss_label)
        optimizer_colorlabels.step()

        if i % 1000 == 0:
            vae_shape_labels.eval()
            vae_color_labels.eval()
            vae.eval()
            # print(labels_color)

            with torch.no_grad():
                # print(color_oneHot[:5])
                # print(z_color_label-z_color)
                # print(loss_of_labels)
                # print('color map for labels')
                # print(z_color_label)

                # print('color map for images')
                # print(z_color)

                recon_imgs = vae.decoder_noskip(z_shape, z_color,0)
                recon_imgs_shape = vae.decoder_shape(z_shape, z_color,0)
                recon_imgs_color = vae.decoder_color(z_shape, z_color,0)

                recon_labels = vae.decoder_noskip(z_shape_label, z_color_label,0)
                recon_shapeOnly = vae.decoder_shape(z_shape_label, 0,0)
                recon_colorOnly = vae.decoder_color(0, z_color_label,0)
                # recon_imgs=vae.decoder_noskip(z_shape)

                sample_size = 20
                orig_imgs = image[:sample_size]
                recon_labels = recon_labels[:sample_size]
                recon_imgs = recon_imgs[:sample_size]
                recon_imgs_shape = recon_imgs_shape[:sample_size]
                recon_imgs_color = recon_imgs_color[:sample_size]
                recon_shapeOnly = recon_shapeOnly[:sample_size]
                recon_colorOnly = recon_colorOnly[:sample_size]

            utils.save_image(
                torch.cat(
                    [orig_imgs, recon_imgs.view(sample_size, 3, 28, 28), recon_imgs_shape.view(sample_size, 3, 28, 28),
                     recon_imgs_color.view(sample_size, 3, 28, 28), recon_labels.view(sample_size, 3, 28, 28),
                     recon_shapeOnly.view(sample_size, 3, 28, 28), recon_colorOnly.view(sample_size, 3, 28, 28)], 0),
                f'sample_training_labels/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

    print(
        '====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_shapelabel / (len(train_loader.dataset) / bs)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch,train_loss_colorlabel / (len(train_loader.dataset) / bs)))