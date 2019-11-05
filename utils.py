import numpy as np

# arbitrary, but we should probably clamp these for faster/better traininig
normalize_max = 50
normalize_min = -70

def normalize_audio(audio):
    return (audio - np.full_like(audio, normalize_min)) / np.full_like(audio, normalize_max - normalize_min)

def unnormalize_audio(audio):
    return (audio * np.full_like(audio, normalize_max - normalize_min)) + np.full_like(audio, normalize_min)

def sample_audio(audio, sample_len=1):
    audio_len = audio.shape[-1]
    assert(audio_len >= sample_len)
    start_idx = np.random.randint(audio_len - sample_len)
    return audio[:,start_idx:start_idx + sample_len]
