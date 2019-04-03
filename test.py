# !/usr/local/bin/python3
import matplotlib
matplotlib.use('agg')

import os
import time
import argparse
import scipy.io

import torch
import pickle
import numpy as np

from datafolder.folder import Test_Dataset
from datafolder.folder import Train_Dataset
from net import *


######################################################################
# Settings
# --------
use_gpu = True
dataset_dict = {
    'market'  :  'Market-1501',
    'duke'  :  'DukeMTMC-reID',
}
model_dict = {
    'resnet18'  :  ResNet18_nFC,
    'resnet34'  :  ResNet34_nFC,
    'resnet50'  :  ResNet50_nFC,
    'densenet'  :  DenseNet121_nFC,
    'resnet50_softmax'  :  ResNet50_nFC_softmax,
}
num_cls_dict = { 'market':30, 'duke':23 }
num_ids_dict = { 'market':751, 'duke':702 }

######################################################################
# Argument
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--data-path', default='/home/litongxin', type=str, help='path to the dataset')
parser.add_argument('--dataset', default='market', type=str, help='dataset')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--batch-size', default=16, type=int, help='batch size')
parser.add_argument('--num-epoch', default=60, type=int, help='num of epoch')
parser.add_argument('--num-workers', default=4, type=int, help='num_workers')
parser.add_argument('--which-epoch',default='last', type=str, help='0,1,2,3...or last')
args = parser.parse_args()

assert args.dataset in dataset_dict.keys()
assert args.model in model_dict.keys()

data_dir = args.data_path
model_dir = os.path.join('./checkpoints', args.dataset, args.model)
result_dir = os.path.join('./result', args.dataset, args.model)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

######################################################################
# Argument
# --------
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
num_cls = num_cls_dict[args.dataset]

######################################################################
# Function
# --------
def load_network(network):
    save_path = os.path.join(model_dir,'net_%s.pth'%args.which_epoch)
    #network.load_state_dict(torch.load(save_path))
    network.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(save_path).items()})
    return network

######################################################################
# Load Data
# ---------
image_datasets = {}

image_datasets['train'] = Train_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                          train_val='train')
image_datasets['query'] = Test_Dataset(data_dir, dataset_name=dataset_dict[args.dataset],
                                       query_gallery='query')
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.num_workers)
              for x in ['train', 'query']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'query']}
labels_name = image_datasets['train'].labels()
######################################################################
# Model
# ---------
model = model_dict[args.model](num_cls)
model = load_network(model)
print(model)
if use_gpu:
    model = model.cuda()
model.train(False)  # Set model to evaluate mode

######################################################################
# Testing
# ------------------
since = time.time()

overall_acc = 0
each_acc = 0
N=dataset_sizes['train']
inter_feature=np.zeros((N,1024))
start=0
# Iterate over data.
for count, data in enumerate(dataloaders['train']):
    # get the inputs
    #images, labels, ids, name = data
    images, ids, labels, id, cam, name = data
    #print(name)
    # wrap them in Variable
    if use_gpu:
        images = images.cuda()
        labels = labels.cuda()
    labels = labels.float()
    # forward
    outputs,outputs_inter = model(images)
    outputs_inter=outputs_inter.data.cpu().numpy()
    #print(outputs_inter.shape)
    #print(outputs_inter[0])
    inter_feature[start:start+args.batch_size,:]=outputs_inter.reshape((-1,1024))
    #inter_feature[start:start+args.batch_size,1024]=name
    start+=args.batch_size
    preds = torch.gt(outputs, torch.ones_like(outputs)/2 ).data
    positive = (preds == labels.data.byte())
    # statistics
    each_acc += torch.sum(positive, dim=0).float()
    running_corrects = torch.sum(positive).item() / labels.size(1)
    overall_acc += running_corrects
    print('step : ({}/{})  |  Acc : {:.4f}'.format(count*args.batch_size, dataset_sizes['train'],
                                                   running_corrects/labels.size(0)))

np.save('/home/litongxin/Person-Attribute-Recognition-MarketDuke/inter_train.npy',inter_feature)
# overall_acc = overall_acc / dataset_sizes['query']
# each_acc = each_acc / dataset_sizes['query']
#
# print('{} Acc: {:.4f}'.format('Overall', overall_acc))
# result = {
#     'overall_acc'   :   overall_acc,
#     'each_acc'      :   each_acc.cpu().numpy(),
#     'labels_name'   :   labels_name,
# }
# scipy.io.savemat(os.path.join(result_dir, 'acc.mat'), result)
#
# time_elapsed = time.time() - since
# print('Testing complete in {:.0f}m {:.0f}s'.format(
#             time_elapsed // 60, time_elapsed % 60))

