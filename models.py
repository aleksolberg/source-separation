import nussl
import torch
from torch import nn
from torch.nn.utils import weight_norm
from nussl.ml.networks.modules import (
    Embedding, DualPath, DualPathBlock, STFT, 
    LearnedFilterBank, AmplitudeToDB, RecurrentStack,
    MelProjection, BatchNorm, InstanceNorm, ShiftAndScale, ConvolutionalStack2D
)


class UNet(nn.Module):
    def __init__(self, num_layers, kernel_size, stride, padding=True, second_layer_channels=16, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = tuple(ti//2 for ti in self.kernel_size) if padding is True else padding
        self.second_layer_channels=second_layer_channels
        if type(dropout) == (float or int):
            self.dropout = [dropout] * self.num_layers
        elif type(dropout) == list:
            '''if len(dropout) != self.num_layers - 1:
                raise IndexError('Dropout list is not of correct length.')
            else:'''
            self.dropout = dropout
        else:
            raise TypeError('Dropout must be either of type \'float\' or \'list\'.')

        # Encoder
        self.convolutional_layers = nn.ModuleList()
        layer = nn.Sequential(
            nn.Conv2d(1, self.second_layer_channels, kernel_size = self.kernel_size, stride=self.stride, padding=self.padding),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.convolutional_layers.append(layer)
        for i in range(self.num_layers - 1):
            layer = nn.Sequential(
                nn.Conv2d(self.second_layer_channels*2**i, self.second_layer_channels*2**(i+1), kernel_size = self.kernel_size, stride=self.stride, padding=self.padding),
                nn.BatchNorm2d(self.second_layer_channels*2**(i+1)),
                nn.LeakyReLU(negative_slope=0.2)
            )
            self.convolutional_layers.append(layer)
        
        # Decoder
        layer = nn.ConvTranspose2d(self.second_layer_channels*2**(self.num_layers-1), self.second_layer_channels*2**(self.num_layers-2), 
                                   kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
        self.deconvolutional_layers = nn.ModuleList()
        self.deconvolutional_layers.append(layer)
        for i in range(self.num_layers - 1, 1, -1):
            layer = nn.ConvTranspose2d(self.second_layer_channels*2**i, self.second_layer_channels*2**(i-2), 
                                       kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
            self.deconvolutional_layers.append(layer)
        layer = nn.ConvTranspose2d(self.second_layer_channels*2, 1, 
                                   kernel_size = self.kernel_size, stride=self.stride, padding=self.padding)
        self.deconvolutional_layers.append(layer)


        self.deconvolutional_BAD_layers = nn.ModuleList()
        for i in range(self.num_layers - 1, 0, -1):
            layer = nn.Sequential(
                nn.BatchNorm2d(self.second_layer_channels*2**(i-1)),
                nn.ReLU(),
                nn.Dropout2d(self.dropout[self.num_layers - (i+1)])
            )
            self.deconvolutional_BAD_layers.append(layer)
        
    def forward(self, data):
        mix_magnitude = data
        data = data.permute(0, 3, 1, 2)

        conv = [data]

        for i in range(self.num_layers):
            data = self.convolutional_layers[i](data)
            conv.append(data)

        data = self.deconvolutional_layers[0](data, output_size = conv[self.num_layers-1].size())
        for i in range(1, self.num_layers-1):
            data = self.deconvolutional_layers[i](torch.cat([data, conv[self.num_layers - i]], 1), output_size = conv[self.num_layers - (i+1)].size())
            data = self.deconvolutional_BAD_layers[i](data)
        
        data = self.deconvolutional_layers[-1](torch.cat([data, conv[1]], 1), output_size = mix_magnitude.permute(0,3,1,2).size())
        mask = torch.sigmoid(data).permute(0, 2, 3, 1).unsqueeze(-1)

        return mask

nussl.ml.register_module(UNet)


def build_UNet(num_layers, kernel_size, stride, padding = True, 
    second_layer_channels=16, dropout=0.5, normalization_class='BatchNorm', 
    normalization_args=None, mix_key='mix_magnitude'):
    
    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
        },
        'mask': {
            'class': 'UNet',
            'args': {
                'num_layers': num_layers,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'second_layer_channels': second_layer_channels,
                'dropout': dropout
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', [mix_key, ]],
        ['normalization', ['log_spectrogram', ]],
        ['mask', ['normalization', ]],
        ['estimates', ['mask', mix_key]]
    ]

    # define the outputs
    output = ['estimates', 'mask']

    # put it together
    config = {
        'name': 'UNet',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_MelUNet(num_layers, kernel_size, stride, num_features, sample_rate, num_mels, padding = True, 
    second_layer_channels=16, dropout=0.5, normalization_class='BatchNorm', 
    normalization_args=None, mix_key='mix_magnitude'):
    
    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
        },
        'mel_projection_forward': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'forward',
                'clamp': True
            }
        },
        'mel_mask': {
            'class': 'UNet',
            'args': {
                'num_layers': num_layers,
                'kernel_size': kernel_size,
                'stride': stride,
                'padding': padding,
                'second_layer_channels': second_layer_channels,
                'dropout': dropout
            }
        },
        'mask': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'backward', 
                'clamp': True
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', [mix_key, ]],
        ['normalization', ['log_spectrogram', ]],
        ['mask', ['normalization', ]],
        ['estimates', ['mask', mix_key]]
    ]
    connections = [
        ['log_spectrogram', [mix_key, ]],
        ['normalization', ['log_spectrogram', ]],
        ['mel_projection_forward', ['normalization', ]],
        ['mel_mask', ['mel_projection_forward', ]],
        ['mask', ['mel_mask', ]],
        ['estimates', ['mask', mix_key]]
    ]

    # define the outputs
    output = ['estimates', 'mask']

    # put it together
    config = {
        'name': 'UNet',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config


def build_recurrent_mask_inference_with_mel_projection(num_features, hidden_size, num_layers, bidirectional, 
    dropout, num_sources, mask_activation, sample_rate, num_mels, 
    num_audio_channels=1, rnn_type='lstm', normalization_class='BatchNorm', 
    normalization_args=None, mix_key='mix_magnitude'):

    normalization_args = {} if normalization_args is None else normalization_args

    # define the building blocks
    modules = {
        mix_key: {},
        'log_spectrogram': {
            'class': 'AmplitudeToDB',
        },
        'normalization': {
            'class': normalization_class,
            'args': normalization_args,
        },
        'mel_projection_forward': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'forward',
                'clamp': True
            }
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': num_mels,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'mel_mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_mels,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels
            }
        },
        'mask': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'backward', 
                'clamp': True
            }
        },
        'estimates': {
            'class': 'Mask',
        },
    }

    # define the topology
    connections = [
        ['log_spectrogram', [mix_key, ]],
        ['normalization', ['log_spectrogram', ]],
        ['mel_projection_forward', ['normalization', ]],
        ['recurrent_stack', ['mel_projection_forward', ]],
        ['mel_mask', ['recurrent_stack', ]],
        ['mask', ['mel_mask', ]],
        ['estimates', ['mask', mix_key]]
    ]

    # define the outputs
    output = ['estimates', 'mask']

    # put it together
    config = {
        'name': 'MelMaskInference',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config