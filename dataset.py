import scaper
import os
import nussl
import numpy as np
import warnings


class MixClosure:
    def __init__(self, fg_folder, bg_folder, event_template, source_labels, incoherent_rate=0.5, duration=5.0, ref_db=-20):
        self.fg_folder = fg_folder
        self.bg_folder = bg_folder
        if not os.path.exists(fg_folder):
            raise OSError('Foreground folder does not exist')

        self.event_template = event_template
        self.sr, self.num_channels = self.get_sr_and_channels()
        self.source_labels = source_labels
        self.incoherent_rate = incoherent_rate
        self.duration = duration
        self.ref_db = ref_db
        
    def __call__(self, dataset, seed):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
        
            # Decide to generate coferent or incoherent mix
            random_state = np.random.RandomState(seed)

            if random_state.rand() > self.incoherent_rate:
                data = self.coherent(seed)
            else:
                data = self.incoherent(seed)
        
        mixture_audio, mixture_jam, annotation_list, stem_audio_list = data
        
        mix = dataset._load_audio_from_array(
            audio_data=mixture_audio, sample_rate=self.sr
        )

        sources = {}
        ann = mixture_jam.annotations.search(namespace='scaper')[0]
        for obs, stem_audio in zip(ann.data, stem_audio_list):
            key = obs.value['label']
            sources[key] = dataset._load_audio_from_array(
                audio_data=stem_audio, sample_rate=self.sr
            )
        
        output = {
            'mix': mix,
            'sources': sources,
            'metadata': mixture_jam
        }
        return output
    
    
    def coherent(self, seed):
        sc = scaper.Scaper(
            duration=self.duration,
            fg_path=str(self.fg_folder),
            bg_path=str(self.bg_folder),
            random_state=seed
        )

        sc.sr = self.sr
        sc.ref_db = self.ref_db
        sc.n_channels = self.num_channels
        
        event_parameters = self.event_template.copy()

        sc.add_event(**event_parameters)
        event = sc._instantiate_event(sc.fg_spec[0])
        sc.reset_fg_event_spec()
        event_parameters['source_time'] = ('const', event.source_time)
        event_parameters['pitch_shift'] = ('const', event.pitch_shift)
        event_parameters['time_stretch'] = ('const', event.time_stretch)


        for label in self.source_labels:
            event_parameters['label'] = ('const', label)
            coherent_source_file = event.source_file.replace(self.event_template['label'][1], label)
            event_parameters['source_file'] = ('const', coherent_source_file)
            sc.add_event(**event_parameters)
        
        return sc.generate(fix_clipping=True)

    def incoherent(self, seed):
        sc = scaper.Scaper(
            duration=self.duration,
            fg_path=str(self.fg_folder),
            bg_path=str(self.bg_folder),
            random_state=seed
        )

        sc.sr = self.sr
        sc.ref_db = self.ref_db
        sc.n_channels = self.num_channels
        
        event_parameters = self.event_template.copy()

        for label in self.source_labels:
            event_parameters['label'] = ('const', label)
            sc.add_event(**event_parameters)
        
        return sc.generate(fix_clipping=True)


    def get_sr_and_channels(self): # Assumes all files in dataset is of same sample rate
        for root, dirs, files in os.walk(self.fg_folder):
            for filename in files:
                if filename.endswith('.wav') or filename.endswith('.mp3'):
                    audio = nussl.AudioSignal(root + '/' + filename)
                    return audio.sample_rate, audio.num_channels