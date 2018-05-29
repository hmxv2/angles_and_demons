import json
import csv
import pandas as pd
import numpy as np
import sys
import pickle
import random
import math
import re

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data

class Net(nn.Module):
    def __init__(self, use_cuda, in_dim, hidden1_dim, hidden2_dim):
        super(Net,self).__init__()
        
        self.use_cuda = use_cuda
        
        self.layer1=nn.Linear(in_dim, hidden1_dim)
        self.layer2=nn.Linear(hidden1_dim, hidden2_dim)
        self.layer3=nn.Linear(hidden2_dim, 2)
        
        self.activate_fun=nn.ReLU()#.Sigmoid()
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(0.5)
        
    def forward(self,x):
        
        if self.use_cuda:
            x=x.cuda()
        
        #print ('1------------    ', x)
        #y=self.dropout(x)
        #y=self.layer1(y)
        y=self.layer1(x)
        y=self.activate_fun(y)
        #print ('2------------    ', y)
        y=self.dropout(y)
        y=self.layer2(y)
        y=self.activate_fun(y)
      
        #print ('3------------    ', y)
        y=self.dropout(y)
        y=self.layer3(y)
        
        y=self.softmax(y)
        return y