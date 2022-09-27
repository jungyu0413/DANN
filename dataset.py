from torchvision import datasets
import torchvision.transforms as transforms
# import package 

def get_dataset(tr_name, te_name, path='/content/data/final_save_data/'):
    # path
    if tr_name in ['A_train', 'A_test', 'B_train', 'B_test']:
        # dataset name
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
            #    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # data preprocessing & augmentation

        tr_dataset = datasets.ImageFolder(path + tr_name + '/', data_transforms['train'])
        # make dataset 
        te_dataset = datasets.ImageFolder(path +  te_name + '/', data_transforms['test'])
        # make dataset

    else:
        raise ValueError('Dataset %s not found!' % tr_name)
    return tr_dataset, te_dataset
    # maker train_dataset, test_dataset