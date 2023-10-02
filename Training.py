#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:04:16 2020

@author: brad
"""
# prerequisites
import torch
import numpy as np
from sklearn import svm
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
import config 

config.init()

from config import numcolors, args
global numcolors
from mVAE import train, test, vae,  thecolorlabels, optimizer, dataset_builder

#define color labels 
#this list of colors is randomly generated at the start of each epoch (down below)

#numcolors indicates where we are in the color sequence 
#this index value is reset back to 0 at the start of each epoch (down below)
numcolors = 0
#this is the amount of variability in r,g,b color values (+/- this amount from the baseline)

#these define the R,G,B color values for each of the 10 colors.  
#Values near the boundaries of 0 and 1 are specified with colorrange to avoid the clipping the random values
'''folder_path = f'output'

if not os.path.exists(folder_path):
    os.mkdir(folder_path)

device ='cuda'
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    vae.train()
    return vae
modelNumber = 1
load_checkpoint('output/checkpoint_threeloss_singlegrad200_smfc.pth'.format(modelNumber=modelNumber))

# reduce sw latent 2-3 dim, quantify sw error, retina error,

bs =100
data_set_flag_single ='sight_word_single'
data_set_flag_sightword ='sightwords'
train_loader_noSkip_single, train_loader_skip_single, test_loader_noSkip_single, test_loader_skip = dataset_builder(data_set_flag_single, bs)
train_loader_noSkip_sightword, train_loader_skip_sightword, test_loader_noSkip_sightword, test_loader_skip = dataset_builder(data_set_flag_sightword, bs)
#torch.save([], 'sightword_acc_data.pt')

for epoch in range(200, 1200):
    #modified to include color labels
    if epoch <= 0:
        train(epoch,'single', train_loader_noSkip_single, train_loader_skip_single, test_loader_noSkip_single)

    elif epoch <= 1000:
        if epoch%5==0:
            train(epoch,'single', train_loader_noSkip_single, train_loader_skip_single, test_loader_noSkip_single)
        else:
            sw_acc, nw_acc = train(epoch,'sightword', train_loader_noSkip_sightword, train_loader_skip_sightword, test_loader_noSkip_sightword)
            accuracy_lst = torch.load('sightword_acc_data.pt')
            accuracy_lst += [sw_acc, nw_acc]
            torch.save(accuracy_lst, 'sightword_acc_data.pt')
            
    torch.cuda.empty_cache()
    colorlabels = np.random.randint(0,10,100000)#regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    #if epoch % 5 == 0:
     #   test('all')

    if epoch in [1,25,50,75,100,150,200,300,400,500,600,700,800,900,1000,1100]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{folder_path}/checkpoint_threeloss_singlegrad{str(epoch)}_smfc.pth')'''

r_data=torch.load('sightword_acc_data.pt')
sw_data, nw_data= [],[]
for i in range(len(r_data)):
    if i%2==0:
        sw_data += [r_data[i]]
    else:
        nw_data += [r_data[i]]

plt.plot(sw_data, label='Sightword Error')
plt.plot(nw_data, label='Nonword Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Sightword Training')
plt.legend()
plt.show()


