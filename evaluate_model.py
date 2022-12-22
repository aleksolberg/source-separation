import nussl
from nussl.datasets import transforms as nussl_tfm
import torch
from models import UNet, MelUNet
import os
import json
import glob
import numpy as np
import utils


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

model_path = 'source-separation/models/MelUNet/01/'
#model = UNet.build(257, 432, 1, 1)

separator = nussl.separation.deep.DeepMaskEstimation(
    nussl.AudioSignal(sample_rate=11025), model_path=model_path + 'checkpoints/best.model.pth',
    device=DEVICE
)

print(separator.config)

seperated_instruments = ['ins3']
other_instruments = ['ins0', 'ins1', 'ins2']

tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([other_instruments]),
])

test_folder = "source-separation/datasets/randomMIDI/PianoViolin11025/WAV/foreground/test"
test_data = nussl.datasets.hooks.MixSourceFolder(folder=test_folder, transform=tfm, stft_params=stft_params)

def evaluate(model_path, test_data, num_save=0):
    songs_saved = 0
    for i in range(len(test_data)):
        item = test_data[i]
        separator.audio_signal = item['mix']
        estimates = separator()

        source_keys = list(item['sources'].keys())
        estimates.append(item['mix'] - estimates[0])
        sources = [item['sources'][k] for k in source_keys]

        evaluator = nussl.evaluation.BSSEvalScale(
            sources, estimates, source_labels=source_keys
        )
        scores = evaluator.evaluate()

        os.makedirs(model_path + 'scores/', exist_ok=True)
        output_file = model_path + 'scores/' + sources[0].file_name.replace('wav', 'json')
        with open(output_file, 'w') as f:
            json.dump(scores, f, indent=4)
        
        if num_save > songs_saved:
            song_name = model_path + 'separated/'+ item['mix'].file_name.replace('.wav', '/')
            os.makedirs(song_name, exist_ok=True)
            estimates[0].write_audio_to_file(song_name + source_keys[0] + '.wav')
            estimates[1].write_audio_to_file(song_name + source_keys[1] + '.wav')
            songs_saved += 1
    estimatesdict = {
        'separated': estimates[0],
        'other': item['mix'] - estimates[0]
    }
    utils.visualize_sources_and_estimates(item['sources'], estimatesdict)
    

    json_files = glob.glob(str(model_path) + '/scores/*.json')
    df = nussl.evaluation.aggregate_score_files(json_files, aggregator=np.nanmedian)
    nussl.evaluation.associate_metrics(separator.model, df, test_data)
    report_card = nussl.evaluation.report_card(df, report_each_source=True)
    print(report_card)
    with open(model_path + 'report_card.json', 'w') as f:
        json.dump(report_card, f, indent=4)

evaluate(model_path, test_data, num_save=5)


