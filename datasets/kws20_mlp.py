###################################################################################################
#
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import errno
import os

import os
import matplotlib.pyplot as plt
import shutil
import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fft
import librosa
import librosa.display
from scipy import signal
import glob
import librosa.util

import numpy as np
import torch
from torch.utils.model_zoo import tqdm
from torchaudio import transforms
import torchaudio
import librosa
import sys



import ai8x

kws20_classes = [
    'up', 'down', 'left', 'right',
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'on', 'off', 'stop', 'go', 'yes', 'no']

class ToMFCC (torch.nn.Module):

    def forward(self, audio):
        mfcc, mfcc_delta, mfcc_delta2=extract_mfcc(audio)
        return mfcc

class Normalize(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args=args

    def forward(self, audio):
        return ai8x.normalize(args=self.args)

class KWSMLP (torch.utils.data.Dataset):

    def __init__ (self, root, transform):
        super().__init__()



def get_kwsmlp_dataset(data, load_train=True, load_test=True):
    (data_dir, args) = data
    root = data_dir
    def_path = os.path.join(root, 'KWS/raw')
    print(def_path)
    is_dir = os.path.isdir(def_path)
    if not is_dir:
        print("******************************************")
        print("Please follow the instructions below:")
        print("Download the dataset to the \'data\' folder by visiting this link: "
              "\'https://www.kaggle.com/datasets/salader/dogs-vs-cats\'")
        print("If you do not have a Kaggle account, sign up first.")
        print("Unzip the downloaded file and find \'test\' and \'train\' folders "
              "and copy them into \'data/cats_vs_dogs\'. ")
        print("Make sure that images are in the following directory structure:")
        print("  \'data/cats_vs_dogs/train/cats\'")
        print("  \'data/cats_vs_dogs/train/dogs\'")
        print("  \'data/cats_vs_dogs/test/cats\'")
        print("  \'data/cats_vs_dogs/test/dogs\'")
        print("Re-run the script. The script will create an \'augmented\' folder ")
        print("with all the original and augmented images. Remove this folder if you want "
              "to change the augmentation and to recreate the dataset.")
        print("******************************************")
        sys.exit("Dataset not found!")
    else:
        processed_dataset_path = os.path.join(root, "KWS/train_test_split")
        if os.path.isdir(processed_dataset_path):
            print("train test split folder exits. Remove if you want to regenerate")

        train_path = os.path.join(processed_dataset_path, "train")
        test_path = os.path.join(processed_dataset_path, "test")

        if not os.path.isdir(processed_dataset_path):
            os.makedirs(processed_dataset_path, exist_ok=True)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(test_path, exist_ok=True)

            # create label folders for training
            for d in kws20_classes:
                mk = os.path.join(train_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise
            
            # create label folders for test
            for d in kws20_classes:
                mk = os.path.join(test_path, d)
                try:
                    os.mkdir(mk)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        print(f'{mk} already exists!')
                    else:
                        raise

            for folders in kws20_classes:
                folder_path=os.path.join(def_path ,folders)
                for file in os.listdir(folder_path):
                    file_path=os.path.join(folder_path, file)
                    temp.append(file_path)
                training_data, testing_data = train_test_split(temp, test_size=0.30, random_state=25)
                for i in training_data:
                    audio, sample_rate = librosa.load(i, sr=16000)
                    audio=librosa.util.fix_length(audio, sample_rate)
                    test = i.split('raw/')
                    detination=test[0] + 'train_test_split/train/'+test[1]
                    scipy.io.wavfile.write(filename=detination, rate=sample_rate, data=np.asarray(audio))
                for i in testing_data:
                    audio, sample_rate = librosa.load(i, sr=16000)
                    audio=librosa.util.fix_length(audio, sample_rate)
                    test = i.split('raw/')
                    detination=test[0] + 'train_test_split/test/'+test[1]
                    scipy.io.wavfile.write(filename=detination, rate=sample_rate, data=np.asarray(audio))
        
        # Loading and normalizing train dataset
        if load_train:
            compose=torch.nn.Sequential(ToMFCC(), Normalize(args=args))
            train_dataset = torchaudio.datasets.AudioFolder(root=train_path,
                                                                transform=compose)
        else:
            train_dataset = None

        # Loading and normalizing test dataset
        if load_test:
            compose=torch.nn.Sequential(ToMFCC(), Normalize(args=args))
            test_dataset = torchaudio.datasets.AudioFolder(root=test_path,
                                                                transform=compose)

            if args.truncate_testset:
                test_dataset.data = test_dataset.data[:1]
        else:
            test_dataset = None

        return train_dataset, test_dataset


datasets = [
    {'name': 'kws20_mlp', 
    'input': (13, 40),
    'output': ('up', 'down', 'left', 'right', 'stop', 'go', 'yes', 'no', 'on', 'off', 'one',
                'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero',
                'UNKNOWN'),
    'weight': (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.07), 
    'loader': get_kwsmlp_dataset,
    },]