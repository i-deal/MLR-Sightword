from torch.utils.data import DataLoader, Subset, ConcatDataset
import torch
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from mVAE import Colorize_func
import random

train_dataset = datasets.MNIST(root='./mnist_data/', train=True,
                               transform=transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)

bs = 5000 # number of unique digits loaded from MNIST
nsw = 5000 # number of copies of each sightword
train_name = 'data/sightwords_MNIST/training.pt'
test_name = 'data/sightwords_MNIST/test.pt'

# make dataset smaller representing each letter at least once in each pos.
sightwords_lst = ['59', '41', '90', '65', '28', '89', '77', '20', '07', '94', '17', '01', '36', '81', '62', '64', '96', '93', '21', '68']
elem_count = len(sightwords_lst)
dataiter_lst = []
# creates a list of tensors representing the mnist digits of the index
for i in range(0, 10):
    cur_indices = [idx for idx, target in enumerate(train_dataset.targets) if target in [i]]
    cur_subset = Subset(train_dataset, cur_indices)
    cur_dataiter = iter(torch.utils.data.DataLoader(dataset=cur_subset, batch_size=bs, shuffle=True,  drop_last= True))
    dataiter_lst += [cur_dataiter.next()]

#make a nonsightwords tensor

# concat tensors in order of x, 9-x or x, x
non_wordlst=[]
for i in range(10):
    for x in range(10):
        if str(i)+str(x) not in sightwords_lst:
            non_wordlst += [str(i)+str(x)]

nonwordfile = open('nonwords.txt','w')
nonwordfile.write(str(non_wordlst))
nonwordfile.close()

total_data = []
for word in tqdm((sightwords_lst)): 
    l_idx = int(word[0])
    r_idx = int(word[1])
    left_data = dataiter_lst[l_idx][0].cpu()
    right_data = dataiter_lst[r_idx][0].cpu()
    for i in range(nsw):
        cur_data = torch.cat((left_data[i], right_data[(len(right_data)-1)-i]), dim=2)
        total_data.append((cur_data, word))

torch.save(total_data, train_name)

test_data = []
for word in tqdm((non_wordlst)): 
    l_idx = int(word[0])
    r_idx = int(word[1])
    left_data = dataiter_lst[l_idx][0].cpu()
    right_data = dataiter_lst[r_idx][0].cpu()
    for i in range(1000): # 1000 of each nonword
        cur_data = torch.cat((left_data[i], right_data[(len(right_data)-1)-i]), dim=2)
        test_data.append((cur_data, word))

torch.save(test_data, test_name)


print('testing')
sample = torch.load(train_name)
utils.save_image(sample[1][0],f'data/sightwords_MNIST/sample_{sample[1][1]}.png')