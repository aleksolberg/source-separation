import torch
import nussl
from nussl.datasets import transforms as nussl_tfm
from dataset import MixClosure
from models import MaskInference, UNet
import utils
from ignite.engine import Events
from ignite.handlers.param_scheduler import LRScheduler

utils.logger()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device', DEVICE)
MAX_MIXTURES = int(1e8) # We'll set this to some impossibly high number for on the fly mixing.
OUTPUT_FOLDER = 'source-separation/models/MaskInference/01'

# Sets parameters of the short time fourier transform done on all audio files
stft_params = nussl.STFTParams(window_length=512, hop_length=128, window_type='sqrt_hann')

seperated_instruments = ['ins3']
other_instruments = ['ins0', 'ins1', 'ins2']

# The preprocessing done on the dataset in order to feed it to the neural network
tfm = nussl_tfm.Compose([
    nussl_tfm.SumSources([other_instruments]),
    nussl_tfm.PhaseSensitiveSpectrumApproximation(),
    nussl_tfm.IndexSources('source_magnitudes', 1),
    nussl_tfm.ToSeparationModel(),
])

# Parameters used in the generation of the dataset using Scaper
template_event_parameters = {
    'label': ('const', 'ins3'),
    'source_file': ('choose', []),
    'source_time': ('uniform', 0, 5),
    'event_time': ('const', 0),
    'event_duration': ('const', 5.0),
    'snr': ('uniform', -5, 5),
    'pitch_shift': ('uniform', -2, 2),
    'time_stretch': ('uniform', 0.8, 1.2)
}

# Paths to the train and validation sets
train_folder = 'source-separation/datasets/randomMIDI/PianoViolin11025/WAV/foreground/train'
val_folder = 'source-separation/datasets/randomMIDI/PianoViolin11025/WAV/foreground/val'

# Path to background folder as Scaper requires it
background_folder = 'source-separation/datasets/randomMIDI/PianoViolin11025/WAV/background'

# Loading the datasets as OnTheFly datasets
train_data = nussl.datasets.OnTheFly(
    stft_params = stft_params,
    transform=tfm,
    num_mixtures=MAX_MIXTURES,
    mix_closure=MixClosure(train_folder, background_folder, template_event_parameters, seperated_instruments + other_instruments)
)
train_dataloader = torch.utils.data.DataLoader(
    train_data, num_workers=1, batch_size=20)

print(train_data[0]['mix_magnitude'].shape)

val_data = nussl.datasets.OnTheFly(
    stft_params = stft_params,
    transform=tfm,
    num_mixtures=20,
    mix_closure=MixClosure(val_folder, background_folder, template_event_parameters, seperated_instruments + other_instruments)
)
val_dataloader = torch.utils.data.DataLoader(
    val_data, num_workers=1, batch_size=20)

#utils.print_existing_modules()
# Defining the model
#nf = stft_params.window_length // 2 + 1
nt, nf, num_channels = train_data[0]['mix_magnitude'].shape
model = nussl.ml.SeparationModel(nussl.ml.networks.builders.build_recurrent_mask_inference(nf, 300, 4, True, 0.2, 1, 'sigmoid')).to(DEVICE)
#model = nussl.ml.SeparationModel(nussl.ml.networks.builders.build_recurrent_chimera(nf, 300, 4, True, 0.2, 20, 'sigmoid', 1, 'sigmoid')).to(DEVICE)
#model = UNet.build(nf, nt, 1, 1).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nussl.ml.train.loss.L1Loss()
torch_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
scheduler = LRScheduler(torch_lr_scheduler)

#utils.print_existing_modules()
#print(model)

def train_step(engine, batch):
    optimizer.zero_grad()
    output = model(batch)
    loss = loss_fn(
        output['estimates'],
        batch['source_magnitudes']
    )
    
    loss.backward()
    optimizer.step()
    
    loss_vals = {
        'L1Loss': loss.item(),
        'loss': loss.item()
    }
    
    return loss_vals


def val_step(engine, batch):
    with torch.no_grad():
        output = model(batch)
    loss = loss_fn(
        output['estimates'],
        batch['source_magnitudes']
    )    
    loss_vals = {
        'L1Loss': loss.item(), 
        'loss': loss.item()
    }
    return loss_vals


if __name__ == '__main__':
    # Create the engines
    trainer, validator = nussl.ml.train.create_train_and_validation_engines(
        train_step, val_step, device=DEVICE
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_lr():
        print('LR:', optimizer.param_groups[0]["lr"])

    # Adding handlers from nussl that print out details about model training run the validation step, and save the models.
    nussl.ml.train.add_stdout_handler(trainer, validator)
    nussl.ml.train.add_validate_and_checkpoint(OUTPUT_FOLDER, model, 
        optimizer, train_data, trainer, val_dataloader, validator)

    trainer.run(
        train_dataloader, 
        epoch_length=20, 
        max_epochs=100
    )