import nussl
import torch
from torch import nn
from torch.nn.utils import weight_norm
from nussl.ml.networks.modules import (
    Embedding, DualPath, DualPathBlock, STFT, 
    LearnedFilterBank, AmplitudeToDB, RecurrentStack,
    MelProjection, BatchNorm, InstanceNorm, ShiftAndScale, ConvolutionalStack2D
)
import utils

class MaskInference(nn.Module):
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
        
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        self.recurrent_stack = RecurrentStack(
            num_features * num_audio_channels, hidden_size, 
            num_layers, bool(bidirectional), dropout
        )
        hidden_size = hidden_size * (int(bidirectional) + 1)
        self.embedding = Embedding(num_features, hidden_size, 
                                   num_sources, activation, 
                                   num_audio_channels)
        
    def forward(self, data):
        mix_magnitude = data # save for masking
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = self.recurrent_stack(data)
        mask = self.embedding(data)
        estimates = mix_magnitude.unsqueeze(-1) * mask

        output = {
            'mask': mask,
            'estimates': estimates
        }
        return output

    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size, 
              num_layers, bidirectional, dropout, num_sources, 
              activation='sigmoid'):
        nussl.ml.register_module(cls)
        modules = {
            'model': {
                'class': 'MaskInference',
                'args': {
                    'num_features': num_features,
                    'num_audio_channels': num_audio_channels,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                    'dropout': dropout,
                    'num_sources': num_sources,
                    'activation': activation
                }
            }
        }
        
        connections = [
            ['model', ['mix_magnitude']]
        ]
        
        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        output = ['estimates', 'mask',]
        config = {
            'name': 'MaskInference',
            'modules': modules,
            'connections': connections,
            'output': output
        }
        return nussl.ml.SeparationModel(config)


class UNet(nn.Module):
    def __init__(self, num_features, num_time, num_sources, num_channels, activation='sigmoid'):
        super().__init__()
        self.amplitude_to_db = AmplitudeToDB()
        self.input_normalization = BatchNorm(num_features)
        # Define the network components
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = (5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(True)
        )
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv1_BAD = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_BAD = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_BAD = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_BAD = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv5 = nn.ConvTranspose2d(64, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_BAD = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )
        self.deconv6 = nn.ConvTranspose2d(32, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)

    def forward(self, data):
        mix_magnitude = data
        
        data = self.amplitude_to_db(mix_magnitude)
        data = self.input_normalization(data)
        data = data.permute(0, 3, 1, 2)

        conv1_out = self.conv1(data)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        deconv1_out = self.deconv1(conv6_out, output_size = conv5_out.size())
        deconv1_out = self.deconv1_BAD(deconv1_out)
        deconv2_out = self.deconv2(torch.cat([deconv1_out, conv5_out], 1), output_size = conv4_out.size())
        deconv2_out = self.deconv2_BAD(deconv2_out)
        deconv3_out = self.deconv3(torch.cat([deconv2_out, conv4_out], 1), output_size = conv3_out.size())
        deconv3_out = self.deconv3_BAD(deconv3_out)
        deconv4_out = self.deconv4(torch.cat([deconv3_out, conv3_out], 1), output_size = conv2_out.size())
        deconv4_out = self.deconv4_BAD(deconv4_out)
        deconv5_out = self.deconv5(torch.cat([deconv4_out, conv2_out], 1), output_size = conv1_out.size())
        deconv5_out = self.deconv5_BAD(deconv5_out)
        deconv6_out = self.deconv6(torch.cat([deconv5_out, conv1_out], 1), output_size = data.size())
        mask = torch.sigmoid(deconv6_out)

        mask = mask.permute(0, 2, 3, 1).unsqueeze(-1)
        estimates = mix_magnitude.unsqueeze(-1) * mask

        output = {
            'mask': mask,
            'estimates': estimates
        }

        return output
    
    @classmethod
    def build(cls, num_features, num_time, num_sources, num_channels, activation='sigmoid'):
        nussl.ml.register_module(cls)
        
        modules = {
            'model': {
                'class': 'UNet',
                'args': {
                    'num_features': num_features,
                    'num_time': num_time, 
                    'num_sources': num_sources, 
                    'num_channels': num_channels, 
                    'activation': activation
                }
            }

        }

        connections = [
            ['model', ['mix_magnitude']]
        ]

        for key in ['mask', 'estimates']:
            modules[key] = {'class': 'Alias'}
            connections.append([key, [f'model:{key}']])
        
        output = ['estimates', 'mask',]

        config = {
            'name': cls.__name__,
            'modules': modules,
            'connections': connections,
            'output': output
        }

        return nussl.ml.SeparationModel(config)


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
                'direction': 'forward'
            }
        },
        'mel_projection_backward': {
            'class': 'MelProjection',
            'args': {
                'sample_rate': sample_rate,
                'num_frequencies': num_features,
                'num_mels': num_mels,
                'direction': 'backward'
            }
        },
        'recurrent_stack': {
            'class': 'RecurrentStack',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'bidirectional': bidirectional,
                'dropout': dropout,
                'rnn_type': rnn_type
            }
        },
        'mask': {
            'class': 'Embedding',
            'args': {
                'num_features': num_features,
                'hidden_size': hidden_size * 2 if bidirectional else hidden_size,
                'embedding_size': num_sources,
                'activation': mask_activation,
                'num_audio_channels': num_audio_channels
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
        ['mel_projection_backward', ['recurrent_stack', ]],
        ['mask', ['mel_projection_backward', ]],
        ['estimates', ['mask', mix_key]]
    ]

    # define the outputs
    output = ['estimates', 'mask']

    # put it together
    config = {
        'name': 'MaskInference',
        'modules': modules,
        'connections': connections,
        'output': output
    }

    return config



'''class RecurrentDPCL():
    def __init__(self, num_features, num_audio_channels, hidden_size,
                 num_layers, bidirectional, dropout, num_sources, 
                activation='sigmoid'):
        super().__init__()
    

    @classmethod
    def build(cls, num_features, num_audio_channels, hidden_size, 
              num_layers, bidirectional, dropout, num_sources, 
              activation='sigmoid'):
        return(nussl.ml.SeparationModel(
            nussl.ml.networks.builders.build_recurrent_dpcl(
                num_features=num_features, 
                hidden_size=hidden_size, 
                num_layers=num_layers, 
                bidirectional=bidirectional,
                dropout=dropout, 
                embedding_size, 
                embedding_activation, 
                num_audio_channels=num_audio_channels))'''