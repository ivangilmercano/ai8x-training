###################################################################################################
#
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
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
"""
Parkinson's Images Dataset
"""
import os

import torchvision
from torchvision import transforms

import ai8x

def get_parkinsons_dataset(data, load_train=True, load_test=True):
    """
        Load the Pakrinson's Images dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "parkinsons")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        ...
