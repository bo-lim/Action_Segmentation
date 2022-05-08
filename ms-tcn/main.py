#!/usr/bin/python2.7

import torch
from batch_gen import BatchGenerator
import os
import argparse
import random
from utils import *
from trainer import *
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
set_seed(seed)

parser = argparse.ArgumentParser()

parser.add_argument('--version', type=int, default=2) # 1 : ms-tcn1, 2: ms-tcn1++
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="50salads") # gtea, 50salads
parser.add_argument('--split', default='1')
parser.add_argument('--dirname', default='output')
parser.add_argument('--wandb_name', default='')
parser.add_argument('--section_num', default=4, type=int)
parser.add_argument('--features_dim', default='2048', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)

parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--num_layers_PG', default =11, type=int)
parser.add_argument('--num_layers_R', default = 10,type=int)
parser.add_argument('--num_R', default=3, type=int)
parser.add_argument('--num_f_maps', default=64, type=int)

args = parser.parse_args()
section_num = args.section_num
version = args.version
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps
features_dim = args.features_dim
bz = args.bz
lr = args.lr
num_epochs = args.num_epochs

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

to_data_path = "../data/"+args.dataset
vid_list_file = to_data_path+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = to_data_path+"/splits/test.split"+args.split+".bundle"
features_path = to_data_path+"/features/"
gt_path = to_data_path+"/groundTruth/"
mapping_file = to_data_path+"/mapping.txt"
model_dir = "./" + args.dirname + "/MS-TCN"+str(version)+"models/"+args.dataset+"/split_"+args.split
results_dir = "./" + args.dirname + "/MS-TCN"+str(version)+"results/"+args.dataset+"/split_"+args.split+"/"+str(num_epochs)+"ep"

mk_dir(model_dir,results_dir)
torch.cuda.empty_cache()
with open(mapping_file, 'r') as f:
    actions = f.read().split('\n')[:-1]
num_classes = len(actions)
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0]) # {'cut_tomato':0,...}
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, section_num, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
trainer = Trainer(args.action, version, section_num, num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim,
                      num_classes, args.dataset, args.split)
for i in range(section_num):
    if args.action == "train":
        wandb.init(project="video_seg", entity="bo-lim", name= args.wandb_name + "ms-tcn" + str(args.version) + '_' + args.split + '_' + str(i) + '/' + str(section_num))
        wandb.config.update(args)

        trainer.train(model_dir, batch_gen, section_num=i, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
