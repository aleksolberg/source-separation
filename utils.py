import nussl
import matplotlib.pyplot as plt
import logging

def visualize_spectrogram(audio_file):
    plt.figure(figsize=(10, 3))
    nussl.utils.visualize_spectrogram(audio_file)
    plt.title(str(audio_file.file_name))
    plt.tight_layout()
    plt.show()

def visualize_sources(source_dict):
    plt.figure(figsize=(10, 3))
    nussl.utils.visualize_sources_as_masks(source_dict)
    #plt.title(str(audio_file.file_name))
    plt.tight_layout()
    plt.show()

def visualize_sources_and_estimates(source_dict, estimates_dict):
    plt.figure(figsize=(10, 6))

    plt.subplot(211)
    plt.title('Sources')
    nussl.utils.visualize_sources_as_masks(source_dict)

    plt.subplot(212)
    plt.title('Estimates')
    nussl.utils.visualize_sources_as_masks(estimates_dict)

    plt.tight_layout()
    plt.show()

def print_existing_modules():
    excluded = ['checkpoint', 'librosa', 'nn', 'np', 'torch', 'warnings']
    print('nussl.ml.modules contents:')
    print('--------------------------')
    existing_modules = [x for x in dir(nussl.ml.modules) if
                        x not in excluded and not x.startswith('__')]
    print('\n'.join(existing_modules))

def logger(level : str = 'info'):
    """
    Logging level to use.

    Parameters
    ----------
    level : str, optional
        Level of logging to use. Choices are 'debug', 
        'info', 'warning', 'error', and 'critical', by 
        default 'info'.
    """
    ALLOWED_LEVELS = ['debug', 'info', 'warning', 'error', 'critical']
    ALLOWED_LEVELS.extend([x.upper() for x in ALLOWED_LEVELS])
    if level not in ALLOWED_LEVELS:
        raise ValueError(f"logging level must be one of {ALLOWED_LEVELS}")
    
    logging.getLogger('sox').setLevel(logging.ERROR)

    level = getattr(logging, level.upper())
    logging.basicConfig(
        format='%(asctime)s | %(filename)s:%(lineno)d %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=level
    )