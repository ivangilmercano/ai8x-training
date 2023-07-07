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
"""
Classes and functions used to create keyword spotting dataset for MLP application.
"""

class KWS20_MLP:
    """
    `SpeechCom v0.02 <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz>`
    Dataset, 1D folded.

    Args:
    root (string): Root directory of dataset where ``KWS/processed/dataset.pt``
        exist.
    classes(array): List of keywords to be used.
    d_type(string): Option for the created dataset. ``train`` or ``test``.
    n_augment(int, optional): Number of augmented samples added to the dataset from
        each sample by random modifications, i.e. stretching, shifting and random noise.
    transform (callable, optional): A function/transform that takes in an PIL image
        and returns a transformed version.
    download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
    save_unquantized (bool, optional): If true, folded but unquantized data is saved.

    """

    url_speechcommand = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
    url_librispeech = 'http://us.openslr.org/resources/12/dev-clean.tar.gz'
    fs = 16000

    class_dict = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5, 
                  'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11,
                  'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'librispeech': 16,
                  'marvin': 17, 'nine': 18, 'no': 19, 'off': 20, 'on': 21, 'one': 22,
                  'right': 23, 'seven': 24, 'sheila': 25, 'six': 26, 'stop': 27,
                  'three': 28, 'tree': 29, 'two': 30, 'up': 31, 'visual': 32, 'wow': 33,
                  'yes': 34, 'zero': 35}

    def __init__(self, root, classes, transform=None, mfcc=None, download=False):
        
        self.root=root
        self.clsses=classes
        self.transform=transform
        self.download=download

        self.__get_mfcc(mfcc)

    @property
    def raw_folder(self):
        """"
        folder directory for raw data
        """
        return os.path.join(self.root, self.__class__.__name__, 'raw')


        



    