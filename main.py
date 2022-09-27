import torch
import train
import model
from utils import get_free_gpu
import utils
import torch.utils.data as torchdata
from dataset import get_dataset
import torch.utils.model_zoo as model_zoo
import numpy as np
# import package and custom python file

save_name = 'omg'

def main():
################# wandb #######################
    # wandb.init(project="dann", name="office-31", entity="jg_lee")
    # can use wandb


################# dataset #####################
###############  upload dataset     ###########
###############################################
    src_trainset, _ = get_dataset('B_train', 'B_test')
    # train, test 별로 image data folder
    tgt_trainset, _ = get_dataset('A_train', 'A_test')
    # train, test 별로 image data folder


################# dataloader ##################
###############  model's input      ###########
###############################################
    source_train_loader = torchdata.DataLoader(src_trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    target_train_loader = torchdata.DataLoader(tgt_trainset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
   

    if torch.cuda.is_available():
        get_free_gpu()
        print('Running GPU : {}'.format(torch.cuda.current_device()))


################# Model #######################
###############   DANN     ####################
###############################################
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        encoder.load_state_dict(torch.load('trained_models(220926-2)/encoder_source_omg.pt'), strict=False)
        classifier.load_state_dict(torch.load('trained_models(220926-2)/classifier_source_omg.pt'), strict=False)


################# Train #######################
###############   ResNet     ##################
###############################################
    #    train.source_only(encoder, classifier, source_train_loader, target_train_loader, save_name)

################# Train #######################
###############  adaptation     ###############
###############################################
        train.dann(encoder, classifier, discriminator, source_train_loader, target_train_loader, save_name)

    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()
# start main fuction