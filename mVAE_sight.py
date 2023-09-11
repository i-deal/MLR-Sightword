
# MNIST VAE from http://github.com/lyeoni/pytorch-mnist-VAE/blob/master/pytorch-mnist-VAE.ipynb 
# Modified by Brad Wyble and Shekoo Hedayati
# Modifications:
# Colorize transform that changes the colors of a grayscale image
# colors are chosen from 10 options:
colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "teal"]
# specified in "colorvals" variable below

# also there is a skip connection from the first layer to the last layer to enable reconstructions of new stimuli
# and the VAE bottleneck is split, having two different maps
# one is trained with a loss function for color only (eliminating all shape info, reserving only the brightest color)
# the other is trained with a loss function for shape only
#

# prerequisites
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
from torch.utils.data import DataLoader, Subset
from config import numcolors, args
from dataloader import notMNIST

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

global colorlabels


colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [colorrange, .5, .5]
]

try:
    import accimage
except ImportError:
    accimage = None


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

# Enter the picture address
# Return tensor variable
def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def Colorize_func_secret(img,npflag = 0):
    global numcolors,colorlabels  # necessary because we will be modifying this counter variable

    thiscolor = colorlabels[numcolors]  # what base color is this?
    thiscolor = np.random.randint(10)

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
      # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

    #img = img.convert('L')

    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    np_img[0,0,0] = thiscolor   #secretely embed the color label inside
        #this is a temporary fix
    #print(np_img[0,0,0])
    #print(numcolors)
    img = Image.fromarray(np_img, 'RGB')
    if npflag ==1:
        img = backup

    return img

def Colorize_func(img):
    global numcolors,colorlabels  

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index

    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange


    #print(img.size)
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')

    return img

#to choose a specific class in case is necessary
def data_filter (data_type, selected_labels):
  data_trans= copy.deepcopy(data_type)
  data_type_labels= data_type.targets
  idx_selected= np.isin(data_type_labels, selected_labels)
  idx_selected=torch.tensor(idx_selected)
  data_trans.targets= data_type_labels[idx_selected]
  data_trans.data = data_type.data[idx_selected]
  return data_trans

def thecolorlabels(datatype):
    
    colornumstart = 0
    coloridx = range(colornumstart, len(datatype))
    labelscolor = colorlabels[coloridx]
    return torch.tensor(labelscolor)

data_set_flag = 'sight_word_single' # mnist, cifar10, padded_mnist, padded_cifar10, sight_word_single

imgsize = 28
retina_size = 56 #by default should be same size as image
vae_type_flag = 'CNN' # must be CNN or FC

# padded cifar10, 2d retina, sight word latent,
def dataset_generator(dataset_name): 
    if dataset_name == 'sight_word_single':
        bs = 100
        class center_in_retina:
            def __init__(self, max_width):
                self.max_width = max_width
                self.pos = torch.zeros((10))
            def __call__(self, img):
                padding_left = (self.max_width-img.size[0])// 2
                padding_right = self.max_width - img.size[0] - padding_left
                padding = (padding_left, 0, padding_right, 0)
                pos = self.pos.clone()
                pos[padding_left//10] = 1
                return ImageOps.expand(img, padding), pos

        class PadAndPosition:
            def __init__(self, transform):
                self.transform = transform
            def __call__(self, img):
                new_img, position = self.transform(img)
                return transforms.ToTensor()(new_img), transforms.ToTensor()(img), position
        
        # Load MNIST datasets, order: [notional_retina, cropped_digit, one-hot_position_vector]   
        train_dataset_single = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                Colorize_func,
                PadAndPosition(center_in_retina(retina_size)),
            ])
        )

        train_loader_single = torch.utils.data.DataLoader(train_dataset_single, shuffle = True, batch_size=bs, drop_last=True)

        test_loader_single = torch.utils.data.DataLoader(train_dataset_single, shuffle = True, batch_size=bs, drop_last=True)

        train_loader_noSkip = train_loader_single
        train_loader_skip = train_loader_single
        test_loader_noSkip = test_loader_single
        test_loader_skip = test_loader_single 

    elif dataset_name == 'padded_mnist':
        bs = 100
        class translate_to_right:
            def __init__(self, max_width):
                self.max_width = max_width
                self.pos = torch.zeros((10))
            def __call__(self, img):
                padding_left = random.randint(self.max_width // 2, self.max_width - img.size[0])
                padding_right = self.max_width - img.size[0] - padding_left
                padding = (padding_left, 0, padding_right, 0)
                pos = self.pos.clone()
                pos[padding_left//10] = 1
                return ImageOps.expand(img, padding), pos

        class translate_to_left:
            def __init__(self, max_width):
                self.max_width = max_width
                self.pos = torch.zeros((10))
            def __call__(self, img):
                padding_left = random.randint(0, (self.max_width // 2) - img.size[0])
                padding_right = self.max_width - img.size[0] - padding_left
                padding = (padding_left, 0, padding_right, 0)
                pos = self.pos.clone()
                pos[padding_left//10] = 1
                return ImageOps.expand(img, padding), pos

        class PadAndPosition:
            def __init__(self, transform):
                self.transform = transform
            def __call__(self, img):
                new_img, position = self.transform(img)
                return transforms.ToTensor()(new_img), transforms.ToTensor()(img), position
        
        # Load MNIST datasets, order: [notional_retina, cropped_digit, one-hot_position_vector]   
        train_dataset_right = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                Colorize_func,
                PadAndPosition(translate_to_right(retina_size)),
            ])
        )

        train_dataset_left = datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                Colorize_func,
                PadAndPosition(translate_to_left(retina_size)),
            ])
        )

        right_indices_0_to_4 = [idx for idx, target in enumerate(train_dataset_right.targets) if target in [0, 1, 2, 3, 4]]
        right_indices_5_to_9 = [idx for idx, target in enumerate(train_dataset_right.targets) if target not in [0, 1, 2, 3, 4]]

        left_indices_5_to_9 = [idx for idx, target in enumerate(train_dataset_left.targets) if target in [5, 6, 7, 8, 9]]
        left_indices_0_to_4 = [idx for idx, target in enumerate(train_dataset_left.targets) if target not in [5, 6, 7, 8, 9]]

        right_subset_0_to_4 = Subset(train_dataset_right, right_indices_0_to_4) #a subset consisting of only digits < 5 and all translated to the right
        left_subset_5_to_9 = Subset(train_dataset_left, left_indices_5_to_9) #a subset consisting of only digits >= 5 and all translated to the left

        right_subset_5_to_9 = Subset(train_dataset_right, right_indices_5_to_9) #a subset consisting of only digits >= 5 and all translated to the right
        left_subset_0_to_4 = Subset(train_dataset_left, left_indices_0_to_4) #a subset consisting of only digits < 5 and all translated to the left

        #combine these subsets to build a set of all digits where digits < 5 are translated to the right and digits >= 5 to the left
        total_train_dataset = right_subset_0_to_4 + left_subset_5_to_9

        #combine these subsets to build a set of all digits where digits < 5 are translated to the left and digits >= 5 to the right
        total_test_dataset = right_subset_5_to_9 + left_subset_0_to_4

        train_loader_total = torch.utils.data.DataLoader(total_train_dataset, shuffle = True, batch_size=bs, drop_last=True)

        test_loader_total = torch.utils.data.DataLoader(total_test_dataset, shuffle = True, batch_size=bs, drop_last=True)

        train_loader_noSkip = train_loader_total
        train_loader_skip = train_loader_total
        test_loader_noSkip = test_loader_total
        test_loader_skip = test_loader_total

    elif dataset_name == 'padded_cifar10':
        bs = 100
        class translate_to_right:
            def __init__(self, max_width):
                self.max_width = max_width
                self.pos = torch.zeros((10))
            def __call__(self, img):
                padding_left = random.randint(self.max_width // 2, self.max_width - img.size[0])
                padding_right = self.max_width - img.size[0] - padding_left
                padding = (padding_left, 0, padding_right, 0)
                pos = self.pos.clone()
                pos[padding_left//10] = 1
                return ImageOps.expand(img, padding), pos

        class translate_to_left:
            def __init__(self, max_width):
                self.max_width = max_width
                self.pos = torch.zeros((10))
            def __call__(self, img):
                padding_left = random.randint(0, (self.max_width // 2) - img.size[0])
                padding_right = self.max_width - img.size[0] - padding_left
                padding = (padding_left, 0, padding_right, 0)
                pos = self.pos.clone()
                pos[padding_left//10] = 1
                return ImageOps.expand(img, padding), pos

        class PadAndPosition:
            def __init__(self, transform):
                self.transform = transform
            def __call__(self, img):
                new_img, position = self.transform(img)
                return transforms.ToTensor()(new_img), transforms.ToTensor()(img), position
        
        # Load MNIST datasets, order: [notional_retina, cropped_digit, one-hot_position_vector]   
        train_dataset_right = datasets.CIFAR10(
            root='./cifar_data/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((imgsize, imgsize)),
                PadAndPosition(translate_to_right(retina_size)),
            ])
        )

        train_dataset_left = datasets.CIFAR10(
            root='./cifar_data/',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((imgsize, imgsize)),
                PadAndPosition(translate_to_left(retina_size)),
            ])
        )

        right_indices_0_to_4 = [idx for idx, target in enumerate(train_dataset_right.targets) if target in [0, 1, 2, 3, 4]]
        #right_indices_5_to_9 = [idx for idx, target in enumerate(train_dataset_right.targets) if target not in [0, 1, 2, 3, 4]]

        left_indices_5_to_9 = [idx for idx, target in enumerate(train_dataset_left.targets) if target in [5, 6, 7, 8, 9]]
        #left_indices_0_to_4 = [idx for idx, target in enumerate(train_dataset_left.targets) if target not in [5, 6, 7, 8, 9]]

        right_subset_0_to_4 = Subset(train_dataset_right, right_indices_0_to_4) #a subset consisting of only digits < 5 and all translated to the right
        left_subset_5_to_9 = Subset(train_dataset_left, left_indices_5_to_9) #a subset consisting of only digits >= 5 and all translated to the left

        #right_subset_5_to_9 = Subset(train_dataset_right, right_indices_5_to_9) #a subset consisting of only digits >= 5 and all translated to the right
        #left_subset_0_to_4 = Subset(train_dataset_left, left_indices_0_to_4) #a subset consisting of only digits < 5 and all translated to the left

        #combine these subsets to build a set of all digits where digits < 5 are translated to the right and digits >= 5 to the left
        total_train_dataset = right_subset_0_to_4 + left_subset_5_to_9

        #combine these subsets to build a set of all digits where digits < 5 are translated to the left and digits >= 5 to the right
        #total_test_dataset = right_subset_5_to_9 + left_subset_0_to_4

        train_loader_total = torch.utils.data.DataLoader(total_train_dataset, shuffle = True, batch_size=bs, drop_last=True)

        #test_loader_total = torch.utils.data.DataLoader(total_test_dataset, shuffle = True, batch_size=bs, drop_last=True)

        train_loader_noSkip = train_loader_total
        train_loader_skip = train_loader_total
        #test_loader_noSkip = test_loader_total
        #test_loader_skip = test_loader_total

    elif dataset_name == 'mnist':
        bs = 100 #batch size
        nw = 1 #number of workers

        # MNIST and Fashion MNIST Datasets
        train_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=True,
                                    transform=transforms.Compose([Colorize_func, transforms.RandomAffine(degrees=0, translate=(0.3, 0)), transforms.ToTensor()]), download=True)

        #transforms.RandomAffine(degrees=0, translate=(0.1, 0)),  # 10% translation in x and y direction
        test_dataset_MNIST = datasets.MNIST(root='./mnist_data/', train=False,
                                    transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=False)

        ftrain_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                                            transform=transforms.Compose([Colorize_func, transforms.ToTensor()]),
                                            download=True)
        ftest_dataset = datasets.FashionMNIST(root='./fashionmnist_data/', train=False,
                                            transform=transforms.Compose([Colorize_func, transforms.ToTensor()]),
                                            download=False)

        train_mnist_labels= train_dataset_MNIST.targets
        ftrain_dataset.targets=ftrain_dataset.targets+ 10
        train_fmnist_labels=ftrain_dataset.targets

        test_mnist_labels= test_dataset_MNIST.targets
        ftest_dataset.targets=ftest_dataset.targets+10
        test_fmnist_label= ftest_dataset.targets

        #skip connection dataset
        train_skip_mnist= datasets.MNIST(root='./mnist_data/', train=True,
                                    transform=transforms.Compose([Colorize_func,transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8), transforms.ToTensor()]), download=True)
        train_skip_fmnist= datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                                            transform=transforms.Compose([Colorize_func, transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8),transforms.ToTensor()]),
                                            download=True)


        train_dataset_skip= torch.utils.data.ConcatDataset((train_skip_mnist ,train_skip_fmnist)) #training skip connection with all images
        #test_dataset_skip= torch.utils.data.ConcatDataset((test_dataset_MNIST ,test_skip_fmnist))

        train_dataset = torch.utils.data.ConcatDataset((train_dataset_MNIST ,ftrain_dataset))
        test_dataset = torch.utils.data.ConcatDataset((test_dataset_MNIST ,ftest_dataset))

        train_loader_noSkip = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)
        train_loader_skip = torch.utils.data.DataLoader(dataset=train_dataset_skip, batch_size=bs, shuffle=True,  drop_last= True)
        test_loader_noSkip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, drop_last=True)
        test_loader_skip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,  drop_last=True)


        #train and test the classifiers on MNIST and f-MNIST
        bs_tr=120000
        bs_te=20000

        train_loader_class= torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs_tr, shuffle=True)
        test_loader_class= torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_te, shuffle=False)

    elif dataset_name == 'cifar10':
        bs = 100
        nw = 8
        bs_tr=60000
        bs_te=10000
        imgsize = 32

        transform = transforms.Compose(
            [transforms.Resize(imgsize),
                transforms.ToTensor(),])

        def dataset_with_indices(cls):
            """
            Modifies the given Dataset class to return a tuple data, target, index
            instead of just data, target.
            """

            def __getitem__(self, index):
                data, target = cls.__getitem__(self, index)
                return data, target, index

            return type(cls.__name__, (cls,), {
            '   __getitem__': __getitem__,})

        #makes a new data set that returns indices
        CIFAR10windicies = dataset_with_indices(datasets.CIFAR10)

        train_dataset = CIFAR10windicies(root='./cifar_data/', train=True ,transform=transform, download=True)
        test_dataset = CIFAR10windicies(root='./cifar_data/', train=False, transform=transform, download=False)

        train_colorlabels = thecolorlabels([bs_tr])
        test_colorlabels = thecolorlabels([bs_te])

        train_shapelabels=train_dataset.targets
        test_shapelabels=test_dataset.targets

        # Data Loader (Input Pipeline)
        train_loader = torch.utils.data.DataLoader(dataset =train_dataset, batch_size=bs, shuffle=True,num_workers = nw)
        train_loader_class = torch.utils.data.DataLoader(dataset =train_dataset, batch_size=bs_tr, shuffle=False,num_workers = nw)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True,num_workers = nw)
        test_loader_class = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs_te, shuffle=False,num_workers = nw)

        train_loader_noSkip = train_loader
        train_loader_skip = train_loader
        test_loader_noSkip = test_loader
        test_loader_skip = test_loader

        #NEW
        #print('Loading the remapped versions of Cifar10')
        #colorremapstrain =torch.from_numpy(np.asarray(torch.load('3clusterstrain.pth'))).permute(0,2,1).type(torch.cuda.FloatTensor)/255
        #colorremapstest =torch.from_numpy(np.asarray(torch.load('3clusterstest.pth'))).permute(0,2,1).type(torch.cuda.FloatTensor)/255

    return train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip, test_loader_class, bs

train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip, test_loader_class, bs = dataset_generator(data_set_flag)

#the modified VAE
class VAE_FC(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE_FC, self).__init__()

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        #decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)

        self.fc7 = nn.Linear(h_dim1, h_dim1)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        hskip = F.relu(self.fc7(h))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), hskip  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_noskip(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def decoder_all(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = (F.relu(self.fc5(h)) + hskip)
        return torch.sigmoid(self.fc6(h))

    def decoder_skip(self, z_shape, z_color, hskip):
        return torch.sigmoid(self.fc6(hskip))

    def forward_layers(self, l1,l2, layernum,whichdecode):
        hskip = F.relu(self.fc7(l1))
        if layernum == 1:

           h = F.relu(self.fc2(l1))
           mu_shape = self.fc31(h)
           log_var_shape = self.fc32(h)
           mu_color = self.fc33(h)
           log_var_color = self.fc34(h)
           z_shape = self.sampling(mu_shape, log_var_shape)
           z_color = self.sampling(mu_color, log_var_color)
        elif layernum==2:
            
            h=l2
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'all'):
            output = self.decoder_all(z_shape, z_color, hskip)
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(z_shape, z_color, hskip)
        else:
            output = self.decoder_noskip(z_shape, z_color, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode, detatchgrad='none'):
        mu_shape, log_var_shape, mu_color, log_var_color, hskip = self.encoder(x.view(-1, 784 * 3))
        if (detatchgrad == 'shape'):
            z_shape = self.sampling(mu_shape, log_var_shape).detach()
        else:
            z_shape = self.sampling(mu_shape, log_var_shape)

        if (detatchgrad == 'color'):
            z_color = self.sampling(mu_color, log_var_color).detach()
        else:
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'all'):
            output = self.decoder_all(z_shape, z_color, hskip)
        elif (whichdecode == 'noskip'):
            output = self.decoder_noskip(z_shape, z_color, 0)
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(0, 0, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color, 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape, 0, 0)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

# modified CNN VAE
class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim):
        super(VAE_CNN, self).__init__()
        # encoder part
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)  # Latent vectors mu and sigma
        #self.fc1 = nn.Linear(int(imgsize / 4) * int(imgsize / 4) * 16, h_dim1)
        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4) * 16, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part
        self.fc31 = nn.Linear(h_dim2, z_dim)  # shape
        self.fc32 = nn.Linear(h_dim2, z_dim)
        self.fc33 = nn.Linear(h_dim2, z_dim)  # color
        self.fc34 = nn.Linear(h_dim2, z_dim)
        self.fc35 = nn.Linear(l_dim, z_dim)  # location
        self.fc36 = nn.Linear(l_dim, z_dim)
        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # shape
        self.fc4c = nn.Linear(z_dim, h_dim2)  # color
        self.fc4l = nn.Linear(z_dim, l_dim)  # location
        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        
        #self.fc5 = nn.Linear(h_dim2, 10000)
        self.fc5l = nn.Linear(l_dim, l_dim)
        #self.fc_bn5 = nn.BatchNorm1d(int(retina_size/4) * int(imgsize/4) * 16)  # Decoder
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)
        # combine recon and location into retina
        self.fc6 = nn.Linear(imgsize+10,(imgsize//2)+(retina_size//2)) # what happens when fc size is increased
        self.fc7 = nn.Linear((imgsize//2)+(retina_size//2), retina_size)
        #self.fc6 = nn.Linear((imgsize+10) * imgsize * 3, imgsize * (retina_size//2) * 3)
        #self.fc7 = nn.Linear(imgsize * (retina_size//2) * 3, retina_size * imgsize * 3)
        self.relu = nn.ReLU()
        self.skipconv = nn.Conv2d(16,16,kernel_size=1,stride=1,padding =0,bias=False)

    def encoder(self, x):
        l = x[2].cuda()
        x = x[1].cuda()
        #save_image(l[0], f'{args.dir}/orig.png')
        h = self.relu(self.bn1(self.conv1(x)))
        hskip = h
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
        h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h), self.fc33(h), self.fc34(h), self.fc35(l), self.fc36(l), hskip # mu, log_var
    
    def sampling_location(self, mu, log_var):
        std = (0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder_retinal(self, z_shape, z_color, z_location, hskip):
        # digit recon
        h = (F.relu(self.fc4c(z_color)) * 2) + (F.relu(self.fc4s(z_shape)) * 1.3)
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).detach().view(-1, 3, imgsize, imgsize) #detach conv
        h = torch.sigmoid(h)
        # location vector recon
        l = F.relu(self.fc4l(z_location))
        l = self.fc5l(l).detach().view(-1,1,1,10) # reshape to concat
        l = torch.sigmoid(l)
        l = l.expand(-1, 3, imgsize, 10) # reshape to concat
        # combine into retina
        h = torch.cat([h,l], dim = 3)
        #b_dim = h.size()[0]
        #h = h.view(b_dim,-1)
        h = self.relu(self.fc6(h))
        h = self.fc7(h).view(-1,3,imgsize,retina_size)
        return torch.sigmoid(h)

    def decoder_color(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4c(z_color))*2
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_shape(self, z_shape, z_color, hskip):
        h = F.relu(self.fc4s(z_shape)) * 1.3
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def decoder_location(self, z_shape, z_color, hskip, z_location):
        h = F.relu(self.fc4l(z_location))
        h = self.fc5l(h)
        return torch.sigmoid(h)

    def decoder_cropped(self, z_shape, z_color, z_location, hskip):
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape))
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)
    '''
    def decoder_skip(self, z_shape, z_color, hskip):
        #h = F.relu(hskip)
        #h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize/4), int(imgsize/4))
        #h = self.relu(self.bn5(self.conv5(h)))
        #h = self.relu(self.bn6(self.conv6(h)))
        #h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(hskip).view(-1, 3, imgsize, retina_size)
        return torch.sigmoid(h)
    '''
    def activations(self, z_shape, z_color, z_location):
        # add location z repr into fc4l
        h = F.relu(self.fc4c(z_color)) + F.relu(self.fc4s(z_shape)) + F.relu(self.fc4l(z_location))
        fc4c = self.fc4c(z_color)
        fc4s = self.fc4s(z_shape)
        fc4l = self.fc4l(z_location)
        fc5 = self.fc5(h)
        return fc4c, fc4s, fc4l, fc5

    def forward_layer1(self, h, whichdecode):  # decode fromo the layer 1 activations
        hskip = F.relu(self.fc7(h))
        h = F.relu(self.fc2(h))
        mu_shape = self.fc31(h)
        log_var_shape = self.fc32(h)
        mu_color = self.fc33(h)
        log_var_color = self.fc34(h)
        z_shape = self.sampling(mu_shape, log_var_shape)
        z_color = self.sampling(mu_color, log_var_color)
        if (whichdecode == 'all'):
            output = self.decoder_all(z_shape, z_color, hskip)
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(z_shape, z_color, hskip)
        else:
            output = self.decoder_noskip(z_shape, z_color, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, whichdecode='noskip', keepgrad=[]):
        mu_shape, log_var_shape, mu_color, log_var_color, mu_location, log_var_location, hskip = self.encoder(x)
        if ('shape' in keepgrad):
            z_shape = self.sampling(mu_shape, log_var_shape)
        else:
            z_shape = self.sampling(mu_shape, log_var_shape).detach()

        if ('color' in keepgrad):
            z_color = self.sampling(mu_color, log_var_color)
        else:
            z_color = self.sampling(mu_color, log_var_color).detach()
    
        if ('location' in keepgrad):
            z_location = self.sampling_location(mu_location, log_var_location)
        else:
            z_location = self.sampling_location(mu_location, log_var_location).detach()
        
        if ('skip' in keepgrad):
            hskip = hskip
        else:
            hskip = hskip.detach()

        if(whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape,z_color, z_location, hskip)
        elif (whichdecode == 'retinal'):
            output = self.decoder_retinal(z_shape,z_color, z_location, 0)
        elif (whichdecode == 'skip'):
            output = self.decoder_skip(0, 0, hskip)
        elif (whichdecode == 'color'):
            output = self.decoder_color(0, z_color , 0)
        elif (whichdecode == 'shape'):
            output = self.decoder_shape(z_shape,0, 0)
        elif (whichdecode == 'location'):
            output = self.decoder_location(0, 0, 0, z_location)

        return output, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location

# build model
def vae_builder(vae_type = vae_type_flag, x_dim = retina_size * imgsize * 3, h_dim1 = 256, h_dim2 = 128, z_dim = 8, l_dim = 10):
    if vae_type_flag == 'FC':
        vae = VAE_FC(x_dim, h_dim1, h_dim2, z_dim)
    else:
        vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim)

    folder_path = f'sample_{vae_type}_{data_set_flag}_smfc'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return vae, z_dim

vae, z_dim = vae_builder()

if torch.cuda.is_available():
    vae.cuda()
    print('CUDA')

optimizer = optim.Adam(vae.parameters())

# return reconstruction error (this will be used to train the skip connection)
def loss_function(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x[0].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, retina_size * imgsize * 3), reduction='sum')
    return BCE 

def loss_function_crop(recon_x, x, mu, log_var, mu_c, log_var_c):
    x = x[1].clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, retina_size * imgsize * 3), reduction='sum')
    return BCE

size1 = imgsize # temporary fix to make adjustments to the loss functions faster

# loss for shape and color

def loss_function_shape(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make grayscale reconstruction
    grayrecon = recon_x.view(bs, 3, imgsize, size1).mean(1)
   
    grayrecon = torch.stack([grayrecon, grayrecon, grayrecon], dim=1)
    # here's a loss BCE based only on the grayscale reconstruction.  Use this in the return statement to kill color learning
    BCEGray = F.binary_cross_entropy(grayrecon.view(-1, size1 * imgsize * 3), x.view(-1,size1 * imgsize * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCEGray + KLD

def loss_function_color(recon_x, x, mu, log_var):
    x = x[1].clone().cuda()
    # make color-only (no shape) reconstruction and use that as the loss function
    recon = recon_x.clone().view(bs, 3, size1 * imgsize)
    # compute the maximum color for the r,g and b channels for each digit separately
    maxr, maxi = torch.max(recon[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(recon[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(recon[:, 2, :], -1, keepdim=True)
    # now build a new reconsutrction that has only the max color, and no shape information at all
    recon[:, 0, :] = maxr
    recon[:, 1, :] = maxg
    recon[:, 2, :] = maxb
    recon = recon.view(-1, size1 * imgsize * 3)
    maxr, maxi = torch.max(x[:, 0, :], -1, keepdim=True)
    maxg, maxi = torch.max(x[:, 1, :], -1, keepdim=True)
    maxb, maxi = torch.max(x[:, 2, :], -1, keepdim=True)
    newx = x.clone()
    newx[:, 0, :] = maxr
    newx[:, 1, :] = maxg
    newx[:, 2, :] = maxb
    newx = newx.view(-1, size1 * imgsize * 3)
    BCE = F.binary_cross_entropy(recon, newx, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def loss_function_location(recon_x, x, mu, log_var):
    x = x[2].clone().cuda()
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train(epoch, whichdecode):
    global numcolors
    colorlabels = np.random.randint(0, 10,1000000)  # regenerate the list of color labels at the start of each test epoch
    numcolors = 0
    vae.train()
    train_loss = 0
    dataiter_noSkip = iter(train_loader_noSkip) #the latent space is trained on MNIST and f-MNIST
    m = 5 # number of seperate training decoders used
    dataiter_skip= iter(train_loader_skip) #The skip connection is trained on notMNIST
    count=0
    loader=tqdm(train_loader_noSkip)
    for i in loader:
        if (whichdecode == 'iterated'):
           if epoch <= 100:
                data = dataiter_noSkip.next()
                data = data[0]
           else:
               data = dataiter_noSkip.next()
               data2 = dataiter_noSkip.next()
               data[0][1] = torch.cat((data[0][1], data2[0][1]), dim=3)
           count += 1
           detachgrad = 'none'
           optimizer.zero_grad()
           if count% m == 0:
                    whichdecode_use = 'shape'
                    keepgrad = ['shape']
            
           elif count% m == 1:
                    whichdecode_use = 'color'
                    keepgrad = ['color']

           elif count% m == 2:
               whichdecode_use = 'location'
               keepgrad = ['location']

           elif count% m == 3:             
               whichdecode_use = 'retinal'
               keepgrad = []

           else:             
               whichdecode_use = 'cropped'
               keepgrad = ['shape', 'color']

        recon_batch, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, whichdecode_use, keepgrad)
        if (whichdecode == 'iterated'):  # if yes, randomly alternate between using and ignoring the skip connections
            if count % m == 0:  # one of out 3 times, let's use the skip connection
                loss = loss_function_shape(recon_batch, data, mu_shape, log_var_shape)  # change the order, was color and shape
                loss.backward()

            elif count% m == 1:
                loss = loss_function_color(recon_batch, data, mu_color, log_var_color)
                loss.backward()
            
            elif count% m == 2:
                loss = loss_function_location(recon_batch, data, mu_location, log_var_location)
                loss.backward()

            elif count% m == 3:
                loss = loss_function(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
                loss.backward()
            
            else:
                loss = loss_function_crop(recon_batch, data, mu_shape, log_var_shape, mu_color, log_var_color)
                loss.backward()

        train_loss += loss.item()
        optimizer.step()
        loader.set_description(
            (
                f'epoch: {epoch}; mse: {loss.item():.5f};'
            )
        )
        sample_data = data
        sample_size = 25
        if count % 500 == 0:
            vae.eval()
            sample_data[0] = sample_data[0][:sample_size]
            sample_data[1] = sample_data[1][:sample_size]
            sample_data[2] = sample_data[2][:sample_size]
            sample = sample_data
            with torch.no_grad():
                reconl, mu_color, log_var_color, mu_shape, log_var_shape,mu_location, log_var_location = vae(sample, 'location') #location
                reconb, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'retinal') #retina
                recond, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'cropped') #digit
                reconc, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'color') #color
                recons, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(sample, 'shape') #shape
            
            empty_retina = torch.zeros((sample_size, 3, imgsize, retina_size))
            save_image(reconl[0:4], f'{args.dir}/orign.png')
            # reconl = remove dim shadows
            n_reconl = empty_retina.clone()
            for i in range(len(reconl)):
                n_reconl[i][0, :, 0:10] = reconl[i]
                n_reconl[i][1, :, 0:10] = reconl[i]
                n_reconl[i][2, :, 0:10] = reconl[i]

            n_recond = empty_retina.clone()
            for i in range(len(recond)):
                n_recond[i][0, :, 0:imgsize] = recond[i][0]
                n_recond[i][1, :, 0:imgsize] = recond[i][1]
                n_recond[i][2, :, 0:imgsize] = recond[i][2]

            n_reconc = empty_retina.clone()
            for i in range(len(reconc)):
                n_reconc[i][0, :, 0:imgsize] = reconc[i][0]
                n_reconc[i][1, :, 0:imgsize] = reconc[i][1]
                n_reconc[i][2, :, 0:imgsize] = reconc[i][2]

            n_recons = empty_retina.clone()
            for i in range(len(recons)):
                n_recons[i][0, :, 0:imgsize] = recons[i][0]
                n_recons[i][1, :, 0:imgsize] = recons[i][1]
                n_recons[i][2, :, 0:imgsize] = recons[i][2]
            n_reconc = n_reconc.cuda()
            n_recons = n_recons.cuda()
            n_reconl = n_reconl.cuda()
            n_recond = n_recond.cuda()
            shape_color_dim = retina_size
            sample = sample[0].cuda()
            utils.save_image(
                torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim), reconb.view(sample_size, 3, imgsize, shape_color_dim), n_recond.view(sample_size, 3, imgsize, shape_color_dim), n_reconl.view(sample_size, 3, imgsize, shape_color_dim),
                           n_reconc.view(sample_size, 3, imgsize, shape_color_dim), n_recons.view(sample_size, 3, imgsize, shape_color_dim)], 0),
                f'sample_{vae_type_flag}_{data_set_flag}_smfc/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png',
                nrow=sample_size,
                normalize=False,
                range=(-1, 1),
            )

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader_noSkip.dataset)))

def test(whichdecode):
    vae.eval()
    global numcolors
    test_loss = 0
    testiter_noSkip = iter(test_loader_noSkip)  # the latent space is trained on MNIST and f-MNIST
    testiter_skip = iter(test_loader_skip)  # The skip connection is trained on notMNIST
    with torch.no_grad():
        for i in range(1, len(test_loader_noSkip)): # get the next batch


            data = testiter_noSkip.next()
            data = data[0]
            recon, mu_color, log_var_color, mu_shape, log_var_shape, mu_location, log_var_location = vae(data, 'noskip')

            # sum up batch loss
            #test_loss += loss_function_shape(recon, data, mu_shape, log_var_shape).item()
            #test_loss += loss_function_color(recon, data, mu_color, log_var_color).item()
            test_loss += loss_function(recon, data, mu_shape, log_var_shape, mu_color, log_var_color).item()

    print('Example reconstruction')
    datac = data[0].cpu()
    datac=datac.view(bs, 3, imgsize, retina_size)
    save_image(datac[0:8], f'{args.dir}/orig.png')
 
    ''' current imagining of shape and color results in random noise
    print('Imagining a shape')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 0
        zs = torch.randn(64, z_dim).cuda() * 1
        zl = torch.randn(64, z_dim).cuda() * 1
        sample = vae.decoder_noskip(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/sampleshape.png')


    print('Imagining a color')
    with torch.no_grad():  # shots off the gradient for everything here
        zc = torch.randn(64, z_dim).cuda() * 1
        zs = torch.randn(64, z_dim).cuda() * 0
        zl = torch.randn(64, z_dim).cuda() * 1
        sample = vae.decoder_noskip(zs, zc, zl, 0).cuda()
        sample=sample.view(64, 3, imgsize, retina_size)
        save_image(sample[0:8], f'{args.dir}/samplecolor.png')
    '''

    test_loss /= len(test_loader_noSkip.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def activations(image):
    l1_act = F.relu(vae.fc1(image))
    l2_act = F.relu(vae.fc2(l1_act))
    mu_shape, log_var_shape, mu_color, log_var_color, hskip = vae.encoder(image)
    shape_act = vae.sampling(mu_shape, log_var_shape)
    color_act = vae.sampling(mu_color, log_var_color)
    return l1_act, l2_act, shape_act, color_act


def activation_fromBP(L1_activationBP, L2_activationBP, layernum):
    if layernum == 1:
        l2_act_bp = F.relu(vae.fc2(L1_activationBP))
        mu_shape = (vae.fc31(l2_act_bp))
        log_var_shape = (vae.fc32(l2_act_bp))
        mu_color = (vae.fc33(l2_act_bp))
        log_var_color = (vae.fc34(l2_act_bp))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    elif layernum == 2:
        mu_shape = (vae.fc31(L2_activationBP))
        log_var_shape = (vae.fc32(L2_activationBP))
        mu_color = (vae.fc33(L2_activationBP))
        log_var_color = (vae.fc34(L2_activationBP))
        shape_act_bp = vae.sampling(mu_shape, log_var_shape)
        color_act_bp = vae.sampling(mu_color, log_var_color)
    return shape_act_bp, color_act_bp


def BP(bp_outdim, l1_act, l2_act, shape_act, color_act, shape_coeff, color_coeff,l1_coeff,l2_coeff, normalize_fact):
    with torch.no_grad():
        bp_in1_dim = l1_act.shape[1]  # dim=512    #inputs to the binding pool
        bp_in2_dim = l2_act.shape[1]  # dim =256
        bp_in3_dim = shape_act.shape[1]  # dim=4
        bp_in4_dim = color_act.shape[1]  # dim=4
        
        #forward weigts from the mVAE layers to the BP
        c1_fw = torch.randn(bp_in1_dim, bp_outdim).cuda()     
        c2_fw = torch.randn(bp_in2_dim, bp_outdim).cuda()
        c3_fw = torch.randn(bp_in3_dim, bp_outdim).cuda()
        c4_fw = torch.randn(bp_in4_dim, bp_outdim).cuda()
        #backward weights from the BP to mVAE layers
        c1_bw = c1_fw.clone().t()
        c2_bw = c2_fw.clone().t()
        c3_bw = c3_fw.clone().t()
        c4_bw = c4_fw.clone().t()
        
        BP_in_all = list()
        shape_out_BP_all = list()
        color_out_BP_all = list()
        BP_layerI_out_all = list()
        BP_layer2_out_all = list()

        for idx in range(l1_act.shape[0]):
            BP_in_eachimg = torch.mm(shape_act[idx, :].view(1, -1), c3_fw) * shape_coeff + torch.mm(
                color_act[idx, :].view(1, -1), c4_fw) * color_coeff  # binding pool inputs (forward activations)
            BP_L1_each = torch.mm(l1_act[idx, :].view(1, -1), c1_fw) * l1_coeff
            BP_L2_each = torch.mm(l2_act[idx, :].view(1, -1), c2_fw) * l2_coeff


            shape_out_eachimg = torch.mm(BP_in_eachimg , c3_bw)  # backward projections from BP to the vae
            color_out_eachimg = torch.mm(BP_in_eachimg , c4_bw)
            L1_out_eachimg = torch.mm(BP_L1_each , c1_bw)
            L2_out_eachimg = torch.mm(BP_L2_each , c2_bw)

            BP_in_all.append(BP_in_eachimg)  # appending and stacking images

            shape_out_BP_all.append(shape_out_eachimg)
            color_out_BP_all.append(color_out_eachimg)
            BP_layerI_out_all.append(L1_out_eachimg)
            BP_layer2_out_all.append(L2_out_eachimg)

            BP_in = torch.stack(BP_in_all)

            shape_out_BP = torch.stack(shape_out_BP_all)
            color_out_BP = torch.stack(color_out_BP_all)
            BP_layerI_out = torch.stack(BP_layerI_out_all)
            BP_layer2_out = torch.stack(BP_layer2_out_all)

            shape_out_BP = shape_out_BP / bp_outdim
            color_out_BP = color_out_BP / bp_outdim
            BP_layerI_out = (BP_layerI_out / bp_outdim ) * normalize_fact
            BP_layer2_out = BP_layer2_out / bp_outdim

#
        return BP_L1_each, shape_out_BP, color_out_BP, BP_layerI_out, BP_layer2_out



def BPTokens(bp_outdim, bpPortion, shape_coef, color_coef, l1_coeff,l2_coeff, shape_act, color_act,l1_act,l2_act, bs_testing, layernum, normalize_fact ):
    # Store and retrieve multiple items in the binding pool
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items 

    with torch.no_grad():  
        notLink_all = list()  # will be used to accumulate the specific token linkages
        BP_in_all = list()  # will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]
        bp_in_L2_dim = l2_act.shape[1]

        shape_out_all = torch.zeros(bs_testing,
                                    bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out_all = torch.zeros(bs_testing,
                                    bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
        L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()

        shape_fw = torch.randn(bp_in_shape_dim,
                               bp_outdim).cuda()  # make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()
        L2_fw = torch.randn(bp_in_L2_dim, bp_outdim).cuda()

        # ENCODING!  Store each item in the binding pool
        for items in range(bs_testing):  # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

            if layernum == 1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)*l1_coeff
            elif layernum==2:
                BP_in_eachimg = torch.mm(l2_act[items, :].view(1, -1), L2_fw)*l2_coeff
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coef + torch.mm(
                    color_act[items, :].view(1, -1), color_fw) * color_coef  # binding pool inputs (forward activations)

            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_all.append(BP_in_eachimg)  # appending and stacking images
            notLink_all.append(notLink)
        # now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items, 1)
        BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # divide by the token percent, as a normalizing factor

        BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
        notLink_all = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens

        # NOW REMEMBER
        for items in range(bs_testing):  # for each item to be retrieved
            BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
            if layernum == 1:
                L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
                L1_out_all[items,:] = (L1_out_eachimg / bpPortion ) * normalize_fact # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
            if layernum==2:

                L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
                L2_out_all[items, :] = L2_out_eachimg / bpPortion  #
            else:
                shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
                color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
                shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
                color_out_all[items, :] = color_out_eachimg / bpPortion

    return shape_out_all, color_out_all, L2_out_all, L1_out_all



def BPTokens_with_labels(bp_outdim, bpPortion,storeLabels, shape_coef, color_coef, shape_act, color_act,l1_act,l2_act,oneHotShape, oneHotcolor, bs_testing, layernum, normalize_fact ):
    # Store and retrieve multiple items including labels in the binding pool 
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items  
  
    with torch.no_grad():  # <---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all = list()  # will be used to accumulate the specific token linkages
        BP_in_all = list()  # will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottleneck
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]
        bp_in_L2_dim = l2_act.shape[1]
        oneHotShape = oneHotShape.cuda()

        oneHotcolor = oneHotcolor.cuda()
        bp_in_Slabels_dim = oneHotShape.shape[1]  # dim =20
        bp_in_Clabels_dim= oneHotcolor.shape[1]


        shape_out_all = torch.zeros(bs_testing,bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out_all = torch.zeros(bs_testing,bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        L1_out_all = torch.zeros(bs_testing, bp_in_L1_dim).cuda()
        L2_out_all = torch.zeros(bs_testing, bp_in_L2_dim).cuda()
        shape_label_out=torch.zeros(bs_testing, bp_in_Slabels_dim).cuda()
        color_label_out = torch.zeros(bs_testing, bp_in_Clabels_dim).cuda()

        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  # make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()
        L2_fw = torch.randn(bp_in_L2_dim, bp_outdim).cuda()
        shape_label_fw=torch.randn(bp_in_Slabels_dim, bp_outdim).cuda()
        color_label_fw = torch.randn(bp_in_Clabels_dim, bp_outdim).cuda()

        # ENCODING!  Store each item in the binding pool
        for items in range(bs_testing):  # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  # list of 0'd BPs for this token

            if layernum == 1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            elif layernum==2:
                BP_in_eachimg = torch.mm(l2_act[items, :].view(1, -1), L2_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw) * shape_coef + torch.mm(color_act[items, :].view(1, -1), color_fw) * color_coef  # binding pool inputs (forward activations)
                BP_in_Slabels_eachimg=torch.mm(oneHotShape [items, :].view(1, -1), shape_label_fw)
                BP_in_Clabels_eachimg = torch.mm(oneHotcolor[items, :].view(1, -1), color_label_fw)


            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_Slabels_eachimg[:, notLink] = 0
            BP_in_Clabels_eachimg[:, notLink] = 0
            if storeLabels==1:
                BP_in_all.append(
                    BP_in_eachimg + BP_in_Slabels_eachimg + BP_in_Clabels_eachimg)  # appending and stacking images
                notLink_all.append(notLink)

            else:
                BP_in_all.append(BP_in_eachimg )  # appending and stacking images
                notLink_all.append(notLink)



        # now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items, 1)
        BP_in_items = torch.sum(BP_in_items, 0).view(1, -1)  # divide by the token percent, as a normalizing factor

        BP_in_items = BP_in_items.repeat(bs_testing, 1)  # repeat the matrix to the number of items to easier retrieve
        notLink_all = torch.stack(notLink_all)  # this is the set of 0'd connections for each of the tokens

        # NOW REMEMBER
        for items in range(bs_testing):  # for each item to be retrieved
            BP_in_items[items, notLink_all[items, :]] = 0  # set the BPs to zero for this token retrieval
            if layernum == 1:
                L1_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L1_fw.t()).cuda()  # do the actual reconstruction
                L1_out_all[items,:] = (L1_out_eachimg / bpPortion ) * normalize_fact # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
            if layernum==2:

                L2_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),L2_fw.t()).cuda()  # do the actual reconstruction
                L2_out_all[items, :] = L2_out_eachimg / bpPortion  #
            else:
                shape_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1),shape_fw.t()).cuda()  # do the actual reconstruction
                color_out_eachimg = torch.mm(BP_in_items[items, :].view(1, -1), color_fw.t()).cuda()
                shapelabel_out_each=torch.mm(BP_in_items[items, :].view(1, -1),shape_label_fw.t()).cuda()
                colorlabel_out_each = torch.mm(BP_in_items[items, :].view(1, -1), color_label_fw.t()).cuda()

                shape_out_all[items, :] = shape_out_eachimg / bpPortion  # put the reconstructions into a bit tensor and then normalize by the effective # of BP nodes
                color_out_all[items, :] = color_out_eachimg / bpPortion
                shape_label_out[items,:]=shapelabel_out_each/bpPortion
                color_label_out[items,:]=colorlabel_out_each/bpPortion

    return shape_out_all, color_out_all, L2_out_all, L1_out_all,shape_label_out,color_label_out


def BPTokens_binding_all(bp_outdim,  bpPortion, shape_coef,color_coef,shape_act,color_act,l1_act,bs_testing,layernum, shape_act_grey, color_act_grey):
    #Store multiple items in the binding pool, then try to retrieve the token of item #1 using its shape as a cue
    # bp_outdim:  size of binding pool
    # bpPortion:  number of binding pool units per token
    # shape_coef:  weight for storing shape information
    # color_coef:  weight for storing shape information
    # shape_act:  activations from shape bottleneck
    # color_act:  activations from color bottleneck
    # bs_testing:   number of items  
    #layernum= either 1 (reconstructions from l1) or 0 (recons from the bottleneck
    with torch.no_grad(): #<---not sure we need this, this code is being executed entirely outside of a training loop
        notLink_all=list()  #will be used to accumulate the specific token linkages
        BP_in_all=list()    #will be used to accumulate the bp activations for each item

        bp_in_shape_dim = shape_act.shape[1]  # neurons in the Bottlenecks
        bp_in_color_dim = color_act.shape[1]
        bp_in_L1_dim = l1_act.shape[1]  # neurons in the Bottleneck
        tokenactivation = torch.zeros(bs_testing)  # used for finding max token
        shape_out = torch.zeros(bs_testing,
                                    bp_in_shape_dim).cuda()  # will be used to accumulate the reconstructed shapes
        color_out= torch.zeros(bs_testing,
                                    bp_in_color_dim).cuda()  # will be used to accumulate the reconstructed colors
        l1_out= torch.zeros(bs_testing, bp_in_L1_dim).cuda()


        shape_fw = torch.randn(bp_in_shape_dim, bp_outdim).cuda()  #make the randomized fixed weights to the binding pool
        color_fw = torch.randn(bp_in_color_dim, bp_outdim).cuda()
        L1_fw = torch.randn(bp_in_L1_dim, bp_outdim).cuda()

        #ENCODING!  Store each item in the binding pool
        for items in range (bs_testing):   # the number of images
            tkLink_tot = torch.randperm(bp_outdim)  # for each token figure out which connections will be set to 0
            notLink = tkLink_tot[bpPortion:]  #list of 0'd BPs for this token
            if layernum==1:
                BP_in_eachimg = torch.mm(l1_act[items, :].view(1, -1), L1_fw)
            else:
                BP_in_eachimg = torch.mm(shape_act[items, :].view(1, -1), shape_fw)+torch.mm(color_act[items, :].view(1, -1), color_fw) # binding pool inputs (forward activations)

            BP_in_eachimg[:, notLink] = 0  # set not linked activations to zero
            BP_in_all.append(BP_in_eachimg)  # appending and stacking images
            notLink_all.append(notLink)

        #now sum all of the BPs together to form one consolidated BP activation set.
        BP_in_items = torch.stack(BP_in_all)
        BP_in_items = torch.squeeze(BP_in_items,1)
        BP_in_items = torch.sum(BP_in_items,0).view(1,-1)   #divide by the token percent, as a normalizing factor

        notLink_all=torch.stack(notLink_all)   # this is the set of 0'd connections for each of the tokens

        retrieve_item = 0
        if layernum==1:
            BP_reactivate = torch.mm(l1_act[retrieve_item, :].view(1, -1), L1_fw)
        else:
            BP_reactivate = torch.mm(shape_act_grey[retrieve_item, :].view(1, -1),shape_fw)  # binding pool retreival
        
        # Multiply the cued version of the BP activity by the stored representations
        BP_reactivate = BP_reactivate  * BP_in_items

        for tokens in range(bs_testing):  # for each token
            BP_reactivate_tok = BP_reactivate.clone()
            BP_reactivate_tok[0,notLink_all[tokens, :]] = 0  # set the BPs to zero for this token retrieval
            # for this demonstration we're assuming that all BP-> token weights are equal to one, so we can just sum the
            # remaining binding pool neurons to get the token activation
            tokenactivation[tokens] = BP_reactivate_tok.sum()

        max, maxtoken =torch.max(tokenactivation,0)   #which token has the most activation

        BP_in_items[0, notLink_all[maxtoken, :]] = 0  #now reconstruct color from that one token
        if layernum==1:

            l1_out = torch.mm(BP_in_items.view(1, -1), L1_fw.t()).cuda() / bpPortion  # do the actual reconstruction
        else:

            shape_out = torch.mm(BP_in_items.view(1, -1), shape_fw.t()).cuda() / bpPortion  # do the actual reconstruction of the BP
            color_out = torch.mm(BP_in_items.view(1, -1), color_fw.t()).cuda() / bpPortion

    return tokenactivation, maxtoken, shape_out,color_out, l1_out 


# defining the classifiers  
clf_ss = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for shape
clf_sc = svm.SVC(C=10, gamma='scale', kernel='rbf')  #classify shape map against color labels
clf_cc = svm.SVC(C=10, gamma='scale', kernel='rbf')  # define the classifier for color
clf_cs = svm.SVC(C=10, gamma='scale', kernel='rbf')#classify color map against shape labels

'''
#training the shape map on shape labels and color labels 
def classifier_shape_train(whichdecode_use):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    train_colorlabels = 0
    numcolors = 0
    vae.eval()
    with torch.no_grad():
            data,train_shapelabels  =next(iter(train_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            train_colorlabels = thecolorlabels(train_dataset)
            print('training shape bottleneck against color labels sc')
            clf_sc.fit(z_shape.cpu().numpy(), train_colorlabels)

            print('training shape bottleneck against shape labels ss')
            clf_ss.fit(z_shape.cpu().numpy(), train_shapelabels)

#testing the shape classifier (one image at a time)
def classifier_shape_test(whichdecode_use, clf_ss, clf_sc,verbose =0):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0
    test_colorlabels=0
    with torch.no_grad():
            data, test_shapelabels= next(iter (test_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_shape = vae.sampling(mu_shape, log_var_shape).cuda()
            test_colorlabels = thecolorlabels(test_dataset)
            pred_ss = torch.tensor(clf_ss.predict(z_shape.cpu()))
            pred_sc = torch.tensor(clf_sc.predict(z_shape.cpu()))

            SSreport = torch.eq(test_shapelabels.cpu(), pred_ss).sum().float() / len(pred_ss)
            SCreport = torch.eq(test_colorlabels.cpu(), pred_sc).sum().float() / len(pred_sc)


            if verbose ==1:
                print('----*************---------shape classification from shape map')
                print(confusion_matrix(test_shapelabels, pred_ss))
                print(classification_report(test_shapelabels, pred_ss))
                print('----************----------color classification from shape map')
                print(confusion_matrix(test_colorlabels, pred_sc))
                print(classification_report(test_colorlabels, pred_sc))
    return pred_ss, pred_sc, SSreport, SCreport

#training the color map on shape and color labels
def classifier_color_train(whichdecode_use):
    vae.eval()
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0
    train_colorlabels = 0
    with torch.no_grad():
            data, train_shapelabels = next(iter (train_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            train_colorlabels = thecolorlabels(train_dataset)
            print('training color bottleneck against color labels cc')
            clf_cc.fit(z_color.cpu().numpy(), train_colorlabels)

            print('training color bottleneck against shape labels cs')
            clf_cs.fit(z_color.cpu().numpy(), train_shapelabels)

#testing the color classifier (one image at a time)
def classifier_color_test(whichdecode_use, clf_cc, clf_cs,verbose=0):
    global colorlabels, numcolors
    colorlabels = np.random.randint(0, 10, 1000000)
    numcolors = 0

    test_colorlabels = 0
    with torch.no_grad():
            data, test_shapelabels = next(iter(test_loader_class))
            data = data.cuda()
            recon_batch, mu_color, log_var_color, mu_shape, log_var_shape = vae(data, whichdecode_use)
       
            z_color = vae.sampling(mu_color, log_var_color).cuda()
            test_colorlabels = thecolorlabels(test_dataset)
            pred_cc = torch.tensor(clf_cc.predict(z_color.cpu()))
            pred_cs = torch.tensor(clf_cs.predict(z_color.cpu()))

            CCreport = torch.eq(test_colorlabels.cpu(), pred_cc).sum().float() / len(pred_cc)
            CSreport = torch.eq(test_shapelabels.cpu(), pred_cs).sum().float() / len(pred_cs)

            if verbose==1:
                print('----**********-------color classification from color map')
                print(confusion_matrix(test_colorlabels, pred_cc))
                print(classification_report(test_colorlabels, pred_cc))


                print('----**********------shape classification from color map')
                print(confusion_matrix(test_shapelabels, pred_cs))
                print(classification_report(test_shapelabels, pred_cs))

    return pred_cc, pred_cs, CCreport, CSreport



#testing on shape for multiple images stored in memory

def classifier_shapemap_test_imgs(shape, shapelabels, colorlabels,numImg, clf_shapeS, clf_shapeC,verbose = 0):

    global numcolors
 
    numImg = int(numImg)

    with torch.no_grad():
        predicted_labels=torch.zeros(1,numImg)
        shape = torch.squeeze(shape, dim=1)
        shape = shape.cuda()
        test_colorlabels = thecolorlabels(test_dataset)
        pred_ssimg = torch.tensor(clf_shapeS.predict(shape.cpu()))
     
        pred_scimg = torch.tensor(clf_shapeC.predict(shape.cpu()))

        SSreport = torch.eq(shapelabels.cpu(), pred_ssimg).sum().float() / len(pred_ssimg)
        SCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_scimg).sum().float() / len(pred_scimg)

        if verbose==1:
            print('----*************---------shape classification from shape map')
            print(confusion_matrix(shapelabels[0:numImg], pred_ssimg))
            print(classification_report(shapelabels[0:numImg], pred_ssimg))
            print('----************----------color classification from shape map')
            print(confusion_matrix(colorlabels[0:numImg], pred_scimg))
            print(classification_report(test_colorlabels[0:numImg], pred_scimg))
    return pred_ssimg, pred_scimg, SSreport, SCreport


#testing on color for multiple images stored in memory
def classifier_colormap_test_imgs(color, shapelabels, colorlabels,numImg, clf_colorC, clf_colorS,verbose = 0):

    
    numImg = int(numImg)


    with torch.no_grad():
      
        color = torch.squeeze(color, dim=1)
        color = color.cuda()
        test_colorlabels = thecolorlabels(test_dataset)


        pred_ccimg = torch.tensor(clf_colorC.predict(color.cpu()))
        pred_csimg = torch.tensor(clf_colorS.predict(color.cpu()))


        CCreport = torch.eq(colorlabels[0:numImg].cpu(), pred_ccimg).sum().float() / len(pred_ccimg)
        CSreport = torch.eq(shapelabels.cpu(), pred_csimg).sum().float() / len(pred_csimg)


        if verbose == 1:
            print('----*************---------color classification from color map')
            print(confusion_matrix(test_colorlabels[0:numImg], pred_ccimg))
            print(classification_report(colorlabels[0:numImg], pred_ccimg))
            print('----************----------shape classification from color map')
            print(confusion_matrix(shapelabels[0:numImg], pred_csimg))
            print(classification_report(shapelabels[0:numImg], pred_csimg))

        return pred_ccimg, pred_csimg, CCreport, CSreport
'''

