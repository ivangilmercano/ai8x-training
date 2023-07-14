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
import hashlib
import os
import tarfile
import time
import urllib
import warnings
from zipfile import ZipFile

import numpy as np
import torch
from torch.utils.model_zoo import tqdm
from torchvision import transforms

import librosa
import pytsmod as tsm
import soundfile as sf

import ai8x

data = '../data/KWS/raw'

def get_kwsmlp_dataset(data, load_train=True, load_test=True):
    (data_dir, data) = data
    path = data_dir
    print(path)

datasets = [
    {'name': 'kws20_mlp', 
    'input': (3, 128, 128), 
    'output': ('cat', 'dog'), 
    'loader': get_kwsmlp_dataset,
    },]


if __name__ == "__main__":
    get_kwsmlp_dataset(data)