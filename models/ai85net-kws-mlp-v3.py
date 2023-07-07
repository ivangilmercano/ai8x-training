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

class AI85KWS20MLPV3(nn.Module):
    """
    Multilayer Perceptron (MLP) application on KWS
    """

    def __init__(self,
                num_classes=21,
                num_channels=512,
                dimensions=(512, 64),
                fc_inputs=7,
                bias=False,
                **kwargs):

        super().__init__()
        self.d1 = ai8x.FusedLinearReLU(num_channels, 512, bias=bias, **kwargs)

        self.d2 = ai8x.FusedLinearReLU(512, 256, bias=bias, **kwargs)

        self.d3 = ai8x.FusedLinearReLU(256, 128, bias=bias, **kwargs)

        self.d4 = ai8x.FusedLinearReLU(128, 256, bias=bias, **kwargs)

        self.d5 = ai8x.FusedLinearReLU(256, 512, bias=bias, **kwargs)

        self.d6 = ai8x.FusedLinearReLU(512, fc_inputs, bias=bias, **kwargs)
        self.d7 = ai8x.Linear(fc_inputs*dimensions[0], num_classes, bias=bias, wide=True,**kwargs)


    def forward(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        x = x.view(x.size(0), -1)
        x = self.d7(x)
        

        return x

def ai85kws20mlpv3(pretrained=False, **kwargs):
    """
    Construct a AI85KWSMLP model.
    rn AI85KWS20MLP(**kwargs)
    """
    assert not pretrained
    return AI85KWS20MLPV3(**kwargs)


"""
Defining the dictionary for the AI85KWSMLP model
"""
models = [
    {
        'name': 'ai85kws20mlpv3',
        'min_input': 1,
        'dim': 1,
    },
]
