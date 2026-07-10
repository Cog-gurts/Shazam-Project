import numpy as np
from microphone import record_audio


def record_and_save(listen_time, file_path):
    frames, sample_rate = record_audio(listen_time)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    array_to_save = np.hstack((sample_rate, samples)).astype(np.int32)
    np.save(file_path, array_to_save)


def load_and_parse(file_path):
    array = np.load(file_path)
    sample_rate = int(array[0])
    samples = array[1:]
    return samples, sample_rate

