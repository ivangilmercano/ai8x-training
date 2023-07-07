###################################################################################################
#
# Copyright (C) 2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Keyword spotting network for AI85/AI86
"""

from torch import nn

import ai8x

class AI85KWS20MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) application on KWS
    """

    def __init__(self,
                num_classes=21,
                num_channels=128,
                dimensions=(128,1),
                bias=False,
                **kwargs):

        super().__init__()
        # self.d1 = ai8x.FusedLinearReLU(num_channels, 256, bias=bias, **kwargs)
        # self.d2 = ai8x.FusedLinearReLU(120, 2, bias=bias, **kwargs)
        # self.d3 = ai8x.Linear(256, num_classes, bias=bias, wide=True,**kwargs)
        self.d1 = ai8x.FusedLinearReLU(num_channels, 64, bias=bias, **kwargs)
        self.d2 = ai8x.FusedLinearReLU(64, 128, bias=bias, **kwargs)
        self.d3 = ai8x.FusedLinearReLU(128, 128, bias=bias, **kwargs)
        self.d4 = ai8x.FusedLinearReLU(128, 64, bias=bias, **kwargs)
        self.d5 = ai8x.FusedLinearReLU(64, 32, bias=bias, **kwargs)
        self.d6 = ai8x.FusedLinearReLU(32, 2, bias=bias, **kwargs)
        self.d7 = ai8x.FusedLinearReLU(96, 100, bias=bias, **kwargs)
        self.d8 = ai8x.FusedLinearReLU(100, 2, bias=bias, **kwargs)
        self.d9 = ai8x.Linear(256, num_classes, bias=bias, wide=True,**kwargs)


    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = self.d7(x)
        x = self.d8(x)
        x = x.view(x.size(0), -1)
        x = self.d9(x)

        return x

def ai85kws20mlp(pretrained=False, **kwargs):
    """
    Construct a AI85KWSMLP model.
    rn AI85KWS20MLP(**kwargs)
    """
    assert not pretrained
    return AI85KWS20MLP(**kwargs)


"""
Defining the dictionary for the AI85KWSMLP model
"""
models = [
    {
        'name': 'ai85kws20mlp',
        'min_input': 1,
        'dim': 1,
    },
]
