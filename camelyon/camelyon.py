import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
import tensorflow as tf

from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader

from datasets import load_dataset, Image, Dataset

from cmi import est_IMLY

from models import model_attributes, LinearReg, ConvNet, FCN, Conv1DClassifier

import argparse

import gc
import torch.nn.functional as F


from wilds.common.data_loaders import get_eval_loader

from wilds import get_dataset

from wilds.common.data_loaders import get_eval_loader




def main():

	num_classes = 2

	parser = argparse.ArgumentParser()

	parser.add_argument('--cmi_reg', default=False, action='store_true')
	parser.add_argument('--epochs', type=int, default=5)
	parser.add_argument('--epochs2', type=int, default=10)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--lr1', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=5e-5)
	parser.add_argument('--reg_st', default=0.5, type=float)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--model_1_pretrained', default=False, action='store_true')
	parser.add_argument('--pretrained', default=False, action='store_true')

	args = parser.parse_args()

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

	dataset = get_dataset(dataset="camelyon17", download=True)

	train_data = dataset.get_subset(
    "train",
    transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
	)

	val_data = dataset.get_subset(
    "val",
    transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
	)

	test_data = dataset.get_subset(
    "test",
    transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
	)

	train_loader = get_train_loader("standard", train_data, batch_size=args.batch_size)
	test_loader = get_eval_loader("standard", test_data, batch_size=args.batch_size)
	val_loader = get_eval_loader("standard", val_data, batch_size = args.batch_size)


	model = torchvision.models.resnet50(pretrained=False)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay)


	d = model.fc.in_features
	model.fc = nn.Linear(d, num_classes)
	model = model.cuda()


	model2 = ConvNet(inputSize = 224, outputSize = 1)
	criterion2 = nn.BCEWithLogitsLoss()
	model2 = model2.cuda()
	

	if (args.cmi_reg):
		if (args.model_1_pretrained):	
			model2 = torchvision.models.densenet121(pretrained=args.pretrained)
			criterion2 = nn.CrossEntropyLoss()
			d = model2.classifier.in_features
			model2.classifier = nn.Linear(d, num_classes)
			model2 = model2.cuda()
			train2(train_loader, model2, criterion2, args.lr1, args.epochs2, test_loader)
		else:
			train2(train_loader, model2, criterion2, args.lr1, args.epochs2, test_loader)
		
		
		print("Starting stage 2 training")
		train_cmi(train_loader, model, model2, criterion, optimizer, args.epochs, args.reg_st, test_loader, val_loader, dataset)

		
		
	
	else:
		train(train_loader, model, criterion, optimizer, args.epochs, test_loader, val_loader, dataset)



def train_cmi(train_loader, model, model2, criterion, optimizer, epochs, reg_st, test_loader, val_loader, dataset):



    model.train()
    with torch.set_grad_enabled(True):
        for j in range(0, epochs):

            all_y_pred = torch.empty(0)
            all_y_true = torch.empty(0)
            all_metadata = torch.empty(0, 4)

            for index, batch in enumerate(train_loader):

                x, y, metadata = batch

                images = x.cuda()
                target = y.cuda()

                outputs = model(images)


                reg_val = mi_reg(model, model2, images, target, torch.device("cuda:0"), None)


                per_sample_losses = criterion(outputs.cuda(), target)
                actual_loss = per_sample_losses.mean()+ reg_st*min((1+j/10),1000)*reg_val


                y_pred = torch.argmax(outputs, dim=1).to('cpu')
                all_y_pred = torch.cat((all_y_pred, y_pred), 0)
                all_y_true = torch.cat((all_y_true, y), 0)
                all_metadata = torch.cat((all_metadata, metadata), 0)

                optimizer.zero_grad()
                actual_loss.backward()
                optimizer.step()


            train_dict = dataset.eval(all_y_pred, all_y_true, all_metadata)
            val_dict = validate(val_loader, model, criterion, dataset)
            test_dict = validate(test_loader, model, criterion, dataset)

            print("Epoch " + str(j) + " train accuracy")
            print(train_dict[0]['acc_avg'])
            print("Epoch " + str(j) + " val accuracy")
            print(val_dict[0]['acc_avg'])
            print("Epoch " + str(j) + " test accuracy")
            print(test_dict[0]['acc_avg'])


def train(train_loader, model, criterion, optimizer, epochs, test_loader, val_loader, dataset):


    model.train().cuda()



    with torch.set_grad_enabled(True):
        for j in range(0, epochs):
            all_y_pred = torch.empty(0)
            all_y_true = torch.empty(0)
            all_metadata = torch.empty(0, 4)
            for index, batch in enumerate(train_loader):	

                x, y, metadata = batch

                data = x.cuda()
                labels = y.cuda()

                outputs = model(data)

                loss = criterion(outputs.squeeze().cuda(), labels)


                y_pred = torch.argmax(outputs, dim=1).to('cpu')
                all_y_pred = torch.cat((all_y_pred, y_pred), 0)
                all_y_true = torch.cat((all_y_true, y), 0)
                all_metadata = torch.cat((all_metadata, metadata), 0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            train_dict = dataset.eval(all_y_pred, all_y_true, all_metadata)
            val_dict = validate(val_loader, model, criterion, dataset)
            test_dict = validate(test_loader, model, criterion, dataset)

            print("Epoch " + str(j) + " train accuracy")
            print(train_dict[0]['acc_avg'])
            print("Epoch " + str(j) + " val accuracy")
            print(val_dict[0]['acc_avg'])
            print("Epoch " + str(j) + " test accuracy")
            print(test_dict[0]['acc_avg'])


def train2(dataloader, model,criterion, lr1, epochs, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr1, weight_decay=1e-4)
    model.train()

    print(len(dataloader))
    for epoch in range(epochs):	
        for index, batch in enumerate(dataloader):

            x, y, metadata = batch

            inputs=x.cuda()
            labels=y.cuda().float()

            outputs = model(inputs)

            loss = criterion(outputs.squeeze().cuda(), labels)
            
            optimizer.zero_grad()
    
            loss.backward()
            optimizer.step()

        
    for params in model.parameters():
        params.requires_grad_(False)




def validate(val_loader, model, criterion, dataset):


    model.eval()

    all_y_pred = torch.empty(0)
    all_y_true = torch.empty(0)
    all_metadata = torch.empty(0, 4)
    
    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            x, y, metadata = batch
            images = x.cuda()
            target = y.cuda()
            output = model(images)

            y_pred = torch.argmax(output, dim=1).to('cpu')
            all_y_pred = torch.cat((all_y_pred, y_pred), 0)
            all_y_true = torch.cat((all_y_true, y), 0)
            all_metadata = torch.cat((all_metadata, metadata), 0)


    return dataset.eval(all_y_pred, all_y_true, all_metadata)


def mi_reg(model, lin, x_train, y_train, device, args):
    I_MLY = est_IMLY(model, lin, x_train, y_train, device, round_=False, args=args)
    return torch.abs(I_MLY)



if __name__ == '__main__':
	main()