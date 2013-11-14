# Frank Chemotti
# fchemotti@gmail.com

import scipy.io.wavfile as wavfile
import numpy as np

def rms(wave):
    '''Return RMS amplitude for a signal.
    
    wave -- numpy array of type int16
    return -- float'''

    return np.sqrt(np.mean((wave.astype('int32') ** 2)))
    
def shuffle(wave, c_size, index):
    '''Return shuffled copy of signal.
    
    wave -- numpy array of type int16, axis 0 should be length at least
           c_size * len(index), can have multiple channels (along axis 1)
    c_size -- int, size of chunks to be shuffled
    index -- list of ints, index[i] = j means chunk i will be moved to chunk j
    return -- numpy array with same shape as wave, contents shuffled
    '''

    wave_out = np.zeros(wave.shape, dtype='int16')
    for index_in, index_out in enumerate(index):
        wave_out[index_out * c_size: (index_out + 1) * c_size] = \
            wave[index_in * c_size: (index_in + 1) * c_size]
    return wave_out
    
def shuffle_faded(wave, c, index, w):
    '''Return shuffled copy of signal, with edges crossfaded.
    
    wave - numpy array of type int16, axis 0 should be length at least
           c_size * len(index), can have multiple channels (along axis 1)
    c -- int, length of chunks to be shuffled
    index -- list of ints, index[i] = j means chunk i will be moved to chunk j
    w -- int, length of crossfaded edge, should be such that w < c
    return -- numpy array with almost same shape as wave, w extra samples are
             added to the end, contents shuffled
    '''

    ch = wave.shape[1]    
    window = np.zeros((c + w, ch), dtype='float')
    window[0:w] = np.linspace(0, 1, w).reshape(w,1) * np.ones((1,ch))
    window[w:c] = np.ones((c-w, ch))
    window[c:c+w] = np.linspace(1, 0, w).reshape(w,1) * np.ones((1,ch))
    extra = np.zeros((w, ch), dtype='int16')
    wave2 = np.append(wave, extra, 0)
    wave_out = np.zeros(wave2.shape, dtype='int16')
    for i_in, i_out in enumerate(index):
        wave_out[i_out * c: i_out * c + c + w] += \
            window * wave2[i_in * c: i_in * c + c + w]
    return wave_out
    

def rms_rank(wave, c_size, desc):
    '''Return list of ranks according to decreasing/increasing rms.
    
    wave -- numpy array of type int16
    c_size -- int, size of chunks to measure and rank
    desc -- True or False for descending or ascending sort
    return -- list of ints, list[i] = j means chunk i is ranked j 
              when sorted by rms'''
    
    n = len(wave) / c_size
    index_rms = [(i , rms(wave[i * c_size : (i+1) * c_size])) 
                 for i in range(n)]
    index_rms.sort(key = lambda x:x[1], reverse=desc)
    return [x[0] for x in index_rms]
    
def chopsort(name, c_size, desc, w):
    '''Writes a wav file generated from name.wav by chopping and sorting.

    name -- string, name of input wave file (without '.wav' extension)
    c_size -- int, size of chunks to chop and sort by
    desc -- True or False for descending or ascending sort
    w -- int, length of crossfaded edges
    result -- writes wave file with shuffled data and modified file name'''
    rate, data = wavfile.read(name + '.wav')
    ranks = rms_rank(data.sum(1), c_size, desc)
    data_sort = shuffle_faded(data, c_size, ranks, w)
    order = 'd' if desc else 'a'
    file_out = name + str(c_size) + order + str(w) + '.wav'
    wavfile.write(file_out, rate, data_sort)

chopsort('cloches07b', 8000, True, 50)
