from cmath import exp
from numpy import source
import torch
import numpy as np
import utils
import torch.optim as optim
import torch.nn as nn
import test
from utils import save_model
from utils import set_model_mode
import torch.utils.data as torchdata
from utils import visualize
from dataset import get_dataset
import torch.nn.functional as F

import params
import torch.utils.model_zoo as model_zoo
# import package and custom python file

# Source : 0, Target :1
_, src_testset = get_dataset('B_train', 'B_test')
_, tgt_testset = get_dataset('A_train', 'A_test')
# make test dataset
source_test_loader = torchdata.DataLoader(src_testset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
target_test_loader = torchdata.DataLoader(tgt_testset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=False)
# make test dataloader

################# Train #######################
###############   ResNet     ##################
###############################################
def source_only(encoder, classifier, source_train_loader, target_train_loader, save_name):
    print("Source-only training")
# resnet feature extractor, label classifier, source data loader, target dataloader, name

################# loss #######################
###############   optim     ##################
##############################################
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    # define loss function
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=0.01, momentum=0.9)
    # optimier
    
    for epoch in range(params.epochs):
      # total epoch
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier])
        # use encoder, classifier

        start_steps = epoch * len(source_train_loader)
        # now start step = now epoch * data length
        # ex) 32, 64, 128, 256 ... 320 if batch_size 32
        total_steps = params.epochs * len(source_train_loader)
        # total step = total epoch * data length
        # ex) 100 * 32 = 3200
################# data #######################
###############   image, label     ###########
############################################## 
        for batch_idx, source_data in enumerate(source_train_loader):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps
            # ex) 32 + (1~num) / 3200
            #     62 + (1~num) / 3200

            source_image, source_label = source_image.cuda(), source_label.cuda()
            optimizer = utils.optimizer_scheduler(epoch=epoch, optimizer=optimizer, p=p)
            optimizer.zero_grad()


################# Model ######################
###############    Input    ##################
##############################################
            source_feature = encoder(source_image)
            # feature extractor output

            class_pred = classifier(source_feature)
            # classification output
            class_loss = classifier_criterion(class_pred, source_label)
            # Classification loss

            class_loss.backward()
            # backpropagation

            optimizer.step()
            # optimizer
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), class_loss.item()))
            # custom print

        if (epoch + 1) % 10 == 0:
            test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='source_only')
    save_model(encoder, classifier, None, 'source', save_name)
  #  visualize(encoder, 'source', save_name)



################# DANN #######################
###############             ##################
##############################################
def dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name):
    print("DANN training")
    count = min(len(source_train_loader), len(target_train_loader))



################# Loss #######################
###############   optim      #################
##############################################
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
    list(encoder.parameters()) +
    list(classifier.parameters()) +
    list(discriminator.parameters()),
    lr=0.01,
    momentum=0.9)
    
    for epoch in range(params.epochs):
        print('Epoch : {}'.format(epoch))
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * count
        total_steps = params.epochs * count
        
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_image, source_label = source_image.cuda(), source_label.cuda()
            # source image
            target_image, target_label = target_image.cuda(), target_label.cuda()
            # target image

############# domain classificaion image ###################
            combined_image = torch.cat((source_image, target_image), 0)
            # concat
            optimizer = utils.optimizer_scheduler(epoch=epoch, optimizer=optimizer, p=p)
            # scheduler
            optimizer.zero_grad()


################# Model ######################
###############   input      #################
##############################################
            combined_feature = encoder(combined_image)
            # 64 x 2048
            source_feature = encoder(source_image)
            # 32 x 2048



################# Model ######################
#########  Classification loss  ##############
##############################################
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)


################# Model ######################
#############  domain loss  ##################
##############################################
            domain_pred = discriminator(combined_feature, alpha)
          #  print(domain_pred)
          #  print('domain_pred : ', domain_pred)
            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
          #  print('domain_source_labels : ', domain_source_labels)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
          #  print('domain_target_labels : ', domain_target_labels)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
          #  print(domain_pred.squeeze())
          #  print(domain_combined_label)
          #  print('domain_combined_label : ', domain_combined_label)
          #  print(domain_combined_label)
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = class_loss + domain_loss
          #  class_loss.backward()
          #  domain_loss.backward()
            total_loss.backward()


            optimizer.step()

            if (batch_idx + 1) % 5 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                batch_idx * len(target_image), count * len(target_image), 100. * batch_idx / count, total_loss.item(), class_loss.item(), domain_loss.item()))

        if (epoch + 1) % 10 == 0:
            test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='dann')

    save_model(encoder, classifier, discriminator, 'source', save_name)
    #visualize(encoder, 'source', save_name)