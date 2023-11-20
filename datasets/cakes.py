###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Cakes Datasets
"""
import errno

import shutil
import sys

import os
import torch
import torchvision
from torchvision import transforms
import ai8x

from PIL import Image

torch.manual_seed(0)

def augment_pad_image(orig_img):
    """
    Augment for padding and resizing
    """
    width, height = orig_img.size
    
    if width > height:
        height_pad = (width - height)/2
        train_transform = transforms.Compose([
            transforms.Pad((0, int(height_pad)), fill=0, padding_mode='constant'),
            transforms.Resize(256),
        ])

        return train_transform(orig_img)

    elif height > width:
        width_pad = (height - width)/2
        train_transform = transforms.Compose([
            transforms.Pad((int(width_pad), 0), fill=0, padding_mode='constant'),
            transforms.Resize(256),
        ])

        return train_transform(orig_img)
    
    else:
        train_transform = transforms.Compose([
            transforms.Resize(256),
        ])

        return train_transform(orig_img)


def augment_affine_jitter_blur(orig_img):

    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
        ])
    return train_transform(orig_img)


def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        ])
    return train_transform(orig_img)

def cakes_get_datasets(data, load_train=True, load_test=True, aug=2):
    """
    Load Cakes Dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "cakes-dataset/")
    train_path = os.path.join(dataset_path + 'train/')
    test_path = os.path.join(dataset_path + 'test/')
    validation_path = os.path.join(dataset_path + 'validation/')
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset to the \'data\' folder by visiting this link: "
              "\'https://www.kaggle.com/datasets/salader/dogs-vs-cats\'")
        print("If you do not have a Kaggle account, sign up first.")
        print("Unzip the downloaded file and find \'test\' and \'train\' folders "
              "and copy them into \'data/cakes-datase\'. ")
        print("Make sure that images are in the following directory structure:")
        print("  \'data/cakes-dataset/train/cats\'")
        print("  \'data/cakes-datase/train/dogs\'")
        print("  \'data/cakes-datase/test/cats\'")
        print("  \'data/cakes-datase/test/dogs\'")
        print("Re-run the script. The script will create an \'augmented\' folder ")
        print("with all the original and augmented images. Remove this folder if you want "
              "to change the augmentation and to recreate the dataset.")
        print("******************************************")
        sys.exit("Dataset not found!")
    else:
        print('yes')
        augmented_dataset = os.path.join(dataset_path, 'augmented')

        if os.path.isdir(augmented_dataset):
            print("augmented folder exits. Remove if you want to regenerate")
        
        # folder and directory creation
        augmented_train_dataset = os.path.join(augmented_dataset, 'train')
        augmented_test_dataset = os.path.join(augmented_dataset, 'test')

        if not os.path.isdir(augmented_dataset):
            os.makedirs(augmented_dataset, exist_ok=True)
            os.makedirs(augmented_train_dataset, exist_ok=True)
            os.makedirs(augmented_test_dataset, exist_ok=True)

            # create label folders
            for d in os.listdir(test_path):
                mk = os.path.join(augmented_test_dataset, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise
            for d in os.listdir(train_path):
                mk = os.path.join(augmented_train_dataset, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise

            # copy test folder files
            test_cnt = 0
            for (dirpath, _, filenames) in os.walk(test_path):
                print(f'copying {dirpath} -> {augmented_test_dataset}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, test_path)
                        destpath = os.path.join(augmented_test_dataset, relsourcepath)

                        destfile = os.path.join(destpath, filename)
                        shutil.copyfile(os.path.join(dirpath, filename), destfile)
                        test_cnt += 1

            # copy and augment train folder files
            train_cnt = 0
            for (dirpath, _, filenames) in os.walk(train_path):
                print(f'copying and augmenting {dirpath} -> {augmented_train_dataset}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, train_path)
                        destpath = os.path.join(augmented_train_dataset, relsourcepath)
                        srcfile = os.path.join(dirpath, filename)
                        destfile = os.path.join(destpath, filename)

                        # original file
                        shutil.copyfile(srcfile, destfile)
                        train_cnt += 1

                        orig_img = Image.open(srcfile)
                        orig_img = orig_img.convert('RGB')
                        
                        #pad and resize
                        padded_img = augment_pad_image(orig_img)

                        # crop center & blur only
                        aug_img = augment_blur(padded_img)
                        augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

                        # random jitter, affine, brightness & blur
                        for i in range(aug):
                            aug_img = augment_affine_jitter_blur(padded_img)
                            augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                            aug_img.save(augfile)
                            train_cnt += 1

            # augment validation and save to train folder
            for (dirpath, _, filenames) in os.walk(validation_path):
                print(f'copying and augmenting {dirpath} -> {augmented_train_dataset}')
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        relsourcepath = os.path.relpath(dirpath, validation_path)
                        destpath = os.path.join(augmented_train_dataset, relsourcepath)
                        srcfile = os.path.join(dirpath, filename)
                        destfile = os.path.join(destpath, filename)

                        # original file
                        shutil.copyfile(srcfile, destfile)
                        train_cnt += 1

                        orig_img = Image.open(srcfile)
                        orig_img = orig_img.convert('RGB')

                        # crop center & blur only
                        aug_img = augment_blur(orig_img)
                        augfile = destfile[:-4] + '_ab' + str(0) + '.jpg'
                        aug_img.save(augfile)
                        train_cnt += 1

                        # random jitter, affine, brightness & blur
                        for i in range(aug):
                            aug_img = augment_affine_jitter_blur(orig_img)
                            augfile = destfile[:-4] + '_aj' + str(i) + '.jpg'
                            aug_img.save(augfile)
                            train_cnt += 1

            print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')
    
 # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=augmented_train_dataset,
                                                         transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=augmented_test_dataset,
                                                        transform=test_transform)

        if args.truncate_testset:
            test_dataset.data = test_dataset.data[:1]
    else:
        test_dataset = None

    return train_dataset, test_dataset


datasets = [
    {
        'name': 'cakes',
        'input': (3, 128, 128),
        'output': ('cheesecake', 'chocolate', 'pie', 'red_velvet'),
        'loader': cakes_get_datasets,
    },
]