import numpy as np
import math

from frequency_filters import *

semis_per_octave = 12

# (height x 1) gate filter for cqt notes
def gate_filter(height, gaussian=False):
    assert(height % 2 == 1)

    if gaussian:
        def f_gaussian(x, mu=0, sig=1):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        
        min_arr, max_arr = -int(height / 2), int(height / 2) + 1
        return np.expand_dims(np.array([f_gaussian(x) for x in range(min_arr, max_arr)]), 1)
    else:
        return np.expand_dims(np.ones(shape=(height)), 1)

# reduces (num_buckets x t x 1) cqt input to (buckets_per_note*octaves x t x 12)
def reduce_cqt(cqt_input, num_buckets=252, buckets_per_note=3, start_note='C1', end_note='B7'):
    freq_map = list(frequency_map().keys())

    start_note_idx, end_note_idx = freq_map.index(start_note), freq_map.index(end_note) + 1
    trimmed_notes = freq_map[start_note_idx : end_note_idx]
    octaves = int((end_note_idx - start_note_idx) / semis_per_octave)
    
    assert(buckets_per_note * semis_per_octave * octaves == num_buckets) # sanity

    result = np.zeros(shape=(buckets_per_note * octaves, cqt_input.shape[1], semis_per_octave))
    
    for idx, note in enumerate(trimmed_notes):        
        octave = int(idx / semis_per_octave)
        note_idx = idx % semis_per_octave

        lower_idx, upper_idx = idx * buckets_per_note, (idx + 1) * buckets_per_note
        lower_octave_idx, upper_octave_idx = octave * buckets_per_note, (octave + 1) * buckets_per_note

        result[lower_octave_idx : upper_octave_idx,:,note_idx] = cqt_input[lower_idx : upper_idx,:]

    return result

def produce_cqt(reduced_cqt, num_buckets=252, buckets_per_note=3, start_note='C1', end_note='B7'):
    freq_map = list(frequency_map().keys())
    start_note_idx, end_note_idx = freq_map.index(start_note), freq_map.index(end_note) + 1
    
    trimmed_notes = freq_map[start_note_idx : end_note_idx]
    
    cqt = np.zeros(shape=(num_buckets, reduced_cqt.shape[1]))
    for idx in range(len(trimmed_notes)):
        semi = idx % semis_per_octave # note_idx
        octave = int(idx / semis_per_octave) # 

        lower_idx, upper_idx = idx * buckets_per_note, (idx + 1) * buckets_per_note
        lower_octave_idx, upper_octave_idx = octave * buckets_per_note, (octave + 1) * buckets_per_note

        cqt[lower_idx:upper_idx,:] = reduced_cqt[lower_octave_idx : upper_octave_idx,:,semi]

    return cqt
