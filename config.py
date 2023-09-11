#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:31:07 2020

@author: brad
"""
import argparse, os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default='output')
parser.add_argument("--z_dim", type=int, default=4)
parser.add_argument("--z_dim2", type=int, default=40)
args = parser.parse_args()

try:
    os.mkdir(args.dir)

except:
    print('Parent Dir already exists')
try:
    os.mkdir(args.dir+'/sample')
except:
    print('Sample Dir already exists')



def init():
    global numcolors, colorlabels
    colorlabels = np.random.randint(0,10,100000)
    numcolors = 0
