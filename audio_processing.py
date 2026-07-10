import numpy as np
import librosa

def samples_to_frequency(samples, sample_rate):

    coeffs = np.fft.rfft(samples) #raw audio samples to frequency coeffs
    amps = np.abs(coeffs) #each freqency strength
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate) #hz frequency from coeffs indexes
    return freqs, amps, coeffs

def extract_samples(filepath):
    audio_array, rate = librosa.load(filepath)
    print(audio_array.shape)


