import numpy as np
import json
import glob
import librosa

import os

# arbitrary, but we should probably clamp these for faster/better traininig
normalize_max = 50
normalize_min = -70

def load_config(config_path):
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        return json.loads(f.read())

def load_audio(path, formats):
    fmts = ['*.{}'.format(f) for f in formats.split(',')]
    files = [p for fmt in fmts for p in glob.glob(os.path.join(path, fmt))]
    return [librosa.load(path) for path in files]

def normalize_audio(audio):
    return (audio - np.full_like(audio, normalize_min)) / np.full_like(audio, normalize_max - normalize_min)

def unnormalize_audio(audio):
    return (audio * np.full_like(audio, normalize_max - normalize_min)) + np.full_like(audio, normalize_min)

def sample_audio(audio, sample_len=1):
    audio_len = audio.shape[-1]
    assert(audio_len >= sample_len)
    start_idx = np.random.randint(audio_len - sample_len)
    return audio[:,start_idx:start_idx + sample_len]
