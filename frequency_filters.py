'''
Gating functions for frequencies
'''

import math
import numpy as np
import re

from functools import lru_cache

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

@lru_cache(maxsize=len(notes))
def frequency_map(root_note='C', start_hz=16.351, start_octave=0, end_octave=8):
    """ Frequency map [note -> frequency] """

    idx = notes.index(root_note)

    adjusted_notes = [notes[n] for n in range(idx, len(notes))] + [notes[n] for n in range(0, idx)]

    freq_map = {}

    semitones = 0
    for octave in range(start_octave, end_octave + 1):
        for note in adjusted_notes:
            freq_map[note + str(octave)] = str(round(start_hz * pow(2, float(semitones/12)), 2))
            semitones += 1

    return freq_map

def gaussian_filter(x, mu=0, sig=1):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

@lru_cache(maxsize=16)
def generate_cqt_frequency_filter(note, num_buckets=252, buckets_per_note=3, start_note='C1', end_note='C8'):
    """
    Gate filter to gate cqt spectrogram through
    Mapping: [ 0 -> floor value, 1 -> pass-thru value ]
    """
    
    assert(buckets_per_note % 2 == 1)

    min_arr, max_arr = -math.floor(buckets_per_note / 2), math.floor(buckets_per_note / 2) + 1
    filter_overlay = np.expand_dims(np.array([gaussian_filter(x) for x in range(min_arr, max_arr)]), 1)

    top = list(frequency_map().keys())

    start_note_idx, end_note_idx = top.index(start_note), top.index(end_note) + 1

    trimmed_notes = top[start_note_idx : end_note_idx]
    
    freq_filter = np.zeros(shape=(num_buckets, 1))

    if note[-1].isdigit():
        start_digit = trimmed_notes.index(note)
        freq_filter[start_digit * buckets_per_note : (start_digit + 1) * buckets_per_note] = filter_overlay 
    else:
        notes = [n for n in trimmed_notes if re.match(note + '[0-9]+', n)]
        start_indices = [trimmed_notes.index(note) for note in notes]
        
        for idx in start_indices:
            # edge case error? maybe invariant is wrong, should revisit. not a big deal rn
            if idx * buckets_per_note == num_buckets:
                break

            freq_filter[idx * buckets_per_note : (idx + 1) * buckets_per_note] = filter_overlay
    
    return freq_filter

def cqt_to_note_map(note, cqt_input, num_buckets=252, buckets_per_note=3, start_note='C1', end_note='C8'):
    if note[-1].isdigit():
        height = buckets_per_note
    else:
        top = list(frequency_map().keys())
        height = int((top.index(end_note) - top.index(start_note)) / 12) * buckets_per_note
    
    result = np.zeros(shape=(cqt_input.shape[0], cqt_input.shape[1], height))

def gate_cqt(note, cqt_input, num_buckets=252, buckets_per_note=3, start_note='C1', end_note='C8'):
    # generate filter for a single cqt slice
    cqt_filter = generate_cqt_frequency_filter(note, num_buckets, buckets_per_note, start_note, end_note)

    # broadcast filter to match cqt_input time dimension
    cqt_filter = np.repeat(cqt_filter, repeats=cqt_input.shape[-1], axis=1)

    minified = np.full(shape=cqt_input.shape, fill_value=np.amin(cqt_input))

    # gate 'cqt_input' by 'cqt_filter'. [1 means let the value through, 0 means minify the value]
    return minified + (cqt_filter * (cqt_input - minified))
