'''
chopsort - shuffles audio in crossfaded chunks, 
           chunk boundaries aligned with attacks
           (attack detection optimized for piano music)

Frank Chemotti
fchemotti@gmail.com
'''
# usage: chopsort_align('dp', 90000)
# shuffles dp.wav using chunk size of 90000 samples


import scipy.io.wavfile as wavfile
import numpy as np
from math import log10, cos, pi

def rms(wave):
    '''Return RMS amplitude for a signal, in dB relative to digital max of 0dB.
    
    wave -- numpy array of type int16, any shape
    return -- float, including possibly -inf'''

    value = np.mean(wave.astype('int32') ** 2)
    if value == 0:
        return float('-inf')
    else:
        return 5 * log10(value) - 45.154499349597181 
        # == 10 * log10(sqrt(value)/32768) 
        # == decibel level of value relative to 32768 (max for 16 bit signed)

def rough_attacks(wave, thresh_db, chunk_size):
    '''Returns a rough list of all times where attacks occur in wave.
    
    wave -- numpy array of audio data
    thresh_db -- minimum increase in RMS level needed to indicate an attack
    chunk_size -- length in samples of chunks over which to measure change
                  in RMS level
    return -- list of rough attack times (in samples), all in multiples of 
              chunk_size (so actual attack may be +/- half chunk_size)
    '''
    rms_chunk = np.zeros((wave.shape[0] / chunk_size), dtype='float')
    for i in range(len(rms_chunk)):
        rms_chunk[i] = rms(wave[i*chunk_size: (i+1)*chunk_size])
    rms_chunk_diff = np.ediff1d(rms_chunk)
    jump_indexes = np.where(rms_chunk_diff > thresh_db)
    return (jump_indexes[0] + 1) * chunk_size

def fine_attack(wave):
    '''Returns time of most significant increase of RMS level in wave.
    
    wave -- numpy array of audio data
    return -- time (with precision of 100 samples) for which increase in RMS 
              from previous 1000 samples to following 1000 is largest
    '''
    fine_size = 100
    chunk_size = 1000 # 23ms = one period of 44hz (at 44100 sample rate)
    fpc = chunk_size / fine_size # fine chunks per chunk
    chunk_size = fpc * fine_size # round down to multiple of fine_size
    fpw = wave.shape[0] / fine_size # fine chunks per entire wave

    rms_chunk = np.zeros((fpw - fpc + 1), dtype='float')
    rms_diff = np.zeros((fpw - 2*fpc + 1), dtype='float')

    for i in range(len(rms_chunk)):
        rms_chunk[i] = rms(wave[i*fine_size : i*fine_size + chunk_size])
    for i in range(len(rms_diff)):
        rms_diff[i] = rms_chunk[i + fpc] - rms_chunk[i]

    return rms_diff.argmax() * fine_size + chunk_size

def find_attacks(wave):
    '''Returns list of times of attacks in wave.
    
    wave -- numpy array of audio data
    return -- list of attack times (in samples)... only intended to precisely 
              locate infrequent and large attacks''' 
    attacks_rough = rough_attacks(wave, 5, 44100)
    attacks_fine = attacks_rough.copy()
    for i in range(len(attacks_rough)):
        start = attacks_rough[i] - 44100
        end = attacks_rough[i] + 44100
        attacks_fine[i] = start + fine_attack(wave[start:end])
    return list(attacks_fine)

def remove_extra_attacks(attacks, chunk_size):
    '''Removes from list those attacks that follow another too closely.
    
    attacks -- sorted ascending list of integers (attack times)
    chunk_size -- minimum difference desired between consecutive attacks
    return -- same list, with extra items removed
    '''
    good_attacks = [attacks[0]]
    for a in attacks:
        if a - good_attacks[-1] >= chunk_size:
            good_attacks.append(a)
    return good_attacks
   
def make_adjusted_chunks(n, chunk_size, attacks):
    '''Makes list of chunk starts/ends, so that attacks occur at boundaries.
    
    n -- number of chunks
    chunk_size -- base size of uniform chunks
    attacks -- list of attacks to which chunk boundaries should be aligned
    return -- numpy array of shape (2,n) holding start/end of each chunk'''
    start = np.array(range(0,n*chunk_size,chunk_size))
    end = start + chunk_size
    for a in attacks:
        i = int((a + float(chunk_size) / 2) / chunk_size)
        if i < n: start[i] = a
        if i > 0: end[i-1] = a
    return np.vstack((start,end)).T

def rms_rank(wave, chunk):
    '''Returns indexes of chunks, sorted by increasing RMS level of wave.
    
    wave -- numpy array of audio data
    chunk -- numpy array of shape (2,n) holding start/end of each chunk
    return -- list of chunk indexes, sorted according to increasing RMS level'''
    index_rms = np.array([rms(wave[c[0]:c[1]]) for c in chunk])
    return index_rms.argsort()

def shuffle_faded(wave, chunk, index, default_size):
    '''Return shuffled copy of signal, with edges crossfaded.
    
    wave -- numpy array of audio data
    chunk -- numpy array of shape (2,n) holding start/end of each chunk
    index -- list of chunk indexes, in order of desired shuffle
    default_size -- size of uniform chunks
    return -- audio data shuffled in chunks according to specified order of 
              indexes, with each chunk fully crossfaded (using audio'''
    f = np.vectorize(lambda x:(0.5 - 0.5 * cos(pi * x)) ** 0.5)
    wave_out = np.zeros(wave.shape, dtype=np.int16)

    #issue: what to do when c[0] = 0 ... or when last_c_len > c[0] ...
    #how to fade up? just let it wrap? I pass.

    d_window_up = f(np.linspace(0,1,default_size).reshape(default_size,1))
    d_window_down = f(np.linspace(1,0,default_size).reshape(default_size,1))


    last_c_len = chunk[index[0],1]-chunk[index[0],0]
    last_out_start = 0
    last_out_end = last_c_len

    for i in index:

        c = chunk[i]

        if last_c_len > c[0]: 
            continue
        
        c_len = c[1] - c[0]
        
        if last_c_len == default_size:
            window_up = d_window_up
        else:
            window_up = f(np.linspace(0,1,last_c_len).reshape(last_c_len,1))
        wave_out[last_out_start:last_out_end] += window_up * wave[c[0]-last_c_len:c[0]]

        if c_len == default_size:
            window_down = d_window_down
        else:
            window_down = f(np.linspace(1,0,c_len).reshape(c_len,1))
        wave_out[last_out_end:last_out_end + c_len] += window_down * wave[c[0]:c[1]]

        last_out_start = last_out_end
        last_out_end += c_len
        last_c_len = c_len
        
    return wave_out
        
def chopsort_align(name, c_size):
    print 'load file...'
    rate, data = wavfile.read(name + '.wav')
    print 'find attacks...'
    attacks = find_attacks(data.mean(1))
    print 'total: ', len(attacks)
    attacks = remove_extra_attacks(attacks, c_size)
    print 'after thinning: ', len(attacks)
    print 'align with attacks...'
    chunk = make_adjusted_chunks(data.shape[0] / c_size, c_size, attacks)
    print 'sort amplitudes...'
    index = rms_rank(data.mean(1), chunk)
    print 'shuffle audio...'
    data_sort = shuffle_faded(data, chunk, index, c_size)
    print 'write output...'
    file_out = name + '-varichunk-' + str(c_size) + '.wav'
    wavfile.write(file_out, rate, data_sort)
    print 'done!'



