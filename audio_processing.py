import numpy as np
import librosa

def samples_to_frequency(samples):
    sample_rate = 44100 #making this a constant instead of a parameter unless if this number will vary
    coeffs = np.fft.rfft(samples) #raw audio samples to frequency coeffs

    #calculating amplitudes
    amps = np.abs(coeffs) #each freqency strength
    N = len(samples)
    amps = np.abs(coeffs) / N
    amps[1 : (-1 if N % 2 == 0 else None)] *= 2
    phases = np.arctan2(-coeffs.imag, coeffs.real)

    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate) #hz frequency from coeffs indexes
    return freqs, amps, phases

def inverse_fft(samples):
    coeffs = np.fft.rfft(samples)
    return np.fft.irfft(coeffs, n=len(samples))

def extract_samples(filepath):
    audio_array, rate = librosa.load(filepath, sr = 44100)
    return audio_array
    
samples = extract_samples("Songs\Michael_Jackson_Billie_Jean.mp3")
freqs, amps, phases = samples_to_frequency(samples)
print(freqs.shape)
print(amps.shape)
print(phases.shape)
print(phases[0:10])
print(np.allclose(inverse_fft(samples), samples))



