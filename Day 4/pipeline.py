import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.morphology import iterate_structure

from typing import Tuple, Callable, List, Dict
from scipy.ndimage import generate_binary_structure
from pathlib import Path
import pyaudio

import librosa
# `recorded_audio` is a numpy array of N audio samples
recorded_audio, sampling_rate = librosa.load("data/trumpet.wav", sr=44100, mono=True)

#%matplotlib inline

from numba import njit

# `@njit` "decorates" the `_peaks` function. This tells Numba to
# compile this function using the "low level virtual machine" (LLVM)
# compiler. The resulting object is a Python function that, when called,
# executes optimized machine code instead of the Python code
# 
# The code used in _peaks adheres strictly to the subset of Python and
# NumPy that is supported by Numba's jit. This is a requirement in order
# for Numba to know how to compile this function to more efficient
# instructions for the machine to execute
@njit
def _peaks(
    data_2d: np.ndarray, nbrhd_row_offsets: np.ndarray, nbrhd_col_offsets: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.
    
    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    nbrhd_row_offsets : numpy.ndarray, shape-(N,)
        The row-index offsets used to traverse the local neighborhood.
        
        E.g., given the row/col-offsets (dr, dc), the element at 
        index (r+dr, c+dc) will reside in the neighborhood centered at (r, c).
    
    nbrhd_col_offsets : numpy.ndarray, shape-(N,)
        The col-index offsets used to traverse the local neighborhood. See
        `nbrhd_row_offsets` for more details.
        
    amp_min : float
        All amplitudes equal to or below this value are excluded from being
        local peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned in 
        column-major order
    """
    peaks = []  # stores the (row, col) locations of all the local peaks

    # Iterating over each element in the the 2-D data 
    # in column-major ordering
    #
    # We want to see if there is a local peak located at
    # row=r, col=c
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            # The amplitude falls beneath the minimum threshold
            # thus this can't be a peak.
            continue
        
        # Iterating over the neighborhood centered on (r, c) to see
        # if (r, c) is associated with the largest value in that
        # neighborhood.
        #
        # dr: offset from r to visit neighbor
        # dc: offset from c to visit neighbor
        for dr, dc in zip(nbrhd_row_offsets, nbrhd_col_offsets):
            if dr == 0 and dc == 0:
                # This would compare (r, c) with itself.. skip!
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                # neighbor falls outside of boundary.. skip!
                continue

            if not (0 <= c + dc < data_2d.shape[1]):
                # neighbor falls outside of boundary.. skip!
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a local peak
            peaks.append((r, c))
    return peaks

# `local_peak_locations` is responsible for taking in the boolean mask `neighborhood`
# and converting it to a form that can be used by `_peaks`. This "outer" code is 
# not compatible with Numba which is why we end up using two functions:
# `local_peak_locations` does some initial pre-processing that is not compatible with
# Numba, and then it calls `_peaks` which contains all of the jit-compatible code
def local_peak_locations(data_2d: np.ndarray, neighborhood: np.ndarray, amp_min: float):
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the specified `amp_min`.
    
    Parameters9
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected
    
    neighborhood : numpy.ndarray, shape-(h, w)
        A boolean mask indicating the "neighborhood" in which each
        datum will be assessed to determine whether or not it is
        a local peak. h and w must be odd-valued numbers
        
    amp_min : float
        All amplitudes at and below this value are excluded from being local 
        peaks.
    
    Returns
    -------
    List[Tuple[int, int]]
        (row, col) index pair for each local peak location, returned
        in column-major ordering.
    
    Notes
    -----
    The local peaks are returned in column-major order, meaning that we 
    iterate over all nbrhd_row_offsets in a given column of `data_2d` in search for
    local peaks, and then move to the next column.
    """

    # We always want our neighborhood to have an odd number
    # of nbrhd_row_offsets and nbrhd_col_offsets so that it has a distinct center element
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1
    
    # Find the indices of the 2D neighborhood where the 
    # values were `True`
    #
    # E.g. (row[i], col[i]) stores the row-col index for
    # the ith True value in the neighborhood (going in row-major order)
    nbrhd_row_indices, nbrhd_col_indices = np.where(neighborhood)
    

    # Shift the neighbor indices so that the center element resides 
    # at coordinate (0, 0) and that the center's neighbors are represented
    # by "offsets" from this center element.
    #
    # E.g., the neighbor above the center will has the offset (-1, 0), and 
    # the neighbor to the right of the center will have the offset (0, 1).
    nbrhd_row_offsets = nbrhd_row_indices - neighborhood.shape[0] // 2
    nbrhd_col_offsets = nbrhd_col_indices - neighborhood.shape[1] // 2

    return _peaks(data_2d, nbrhd_row_offsets, nbrhd_col_offsets, amp_min=amp_min)

def local_peaks_mask(data: np.ndarray, cutoff: float) -> np.ndarray:
    """Find local peaks in a 2D array of data and return a 2D array
    that is 1 wherever there is a peak and 0 where there is not.

    Parameters
    ----------
    data : numpy.ndarray, shape-(H, W)

    cutoff : float
         A threshold value that distinguishes background from foreground

    Returns
    -------
    Binary indicator, of the same shape as `data`. The value of
    1 indicates a local peak."""
    # Generate a rank-2, connectivity-2 neighborhood array
    # We will not use `iterate_structure` in this example
    neighborhood_array = generate_binary_structure(rank=2, connectivity=2)

    # Use that neighborhood to find the local peaks in `data`.
    # Pass `cutoff` as `amp_min` to `local_peak_locations`.
    peak_locations = local_peak_locations(data, neighborhood_array, amp_min=cutoff)

    # Turns the list of (row, col) peak locations into a shape-(N_peak, 2) array
    # Save the result to the variable `peak_locations`
    peak_locations = np.array(peak_locations)

    # create a boolean mask of zeros with the same shape as `data`
    mask = np.zeros(data.shape, dtype=bool)

    # populate the local peaks with `1`
    mask[peak_locations[:, 0], peak_locations[:, 1]] = 1
    return mask

def plot_compare(
    data: np.ndarray,
    peak_rendering_func: Callable[[np.ndarray], np.ndarray],
    cutoff: float = -np.inf,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot the original data side-by-side with the binary indicator
    for the local peaks.

    Parameters
    ----------
    data : numpy.ndarray, shape=(N, H, W)
        N 2D arrays of shape (H, W)

    peak_finding_function : Callable[[ndarray], ndarray]
        A function that will locate the 2D peaks in `data` and
        create an image with the 2D peaks 

    cutoff : float, optional (default=-np.inf)
         A threshold value that distinguishes background from foreground
         
    Returns
    -------
    Tuple[matplotlib.Figure, matplotlib.Axes]
        The figure and axes objects of the plot
    """
    fig, ax = plt.subplots(nrows=len(data), ncols=2)
    for i, dat in enumerate(data):
        ax[i, 0].imshow(dat)
        ax[i, 1].imshow(peak_rendering_func(dat, cutoff=cutoff))
    return fig, ax

# running this cell loads the PCM-encoded data for the trumpet clip
import librosa

trumpet_audio, sampling_rate = librosa.load("data/trumpet.wav", sr=44100, mono=True)


# using matplotlib's built-in spectrogram function
fig, ax = plt.subplots()

S, freqs, times, im = ax.specgram(
    trumpet_audio,
    NFFT=4096,
    Fs=sampling_rate,
    window=mlab.window_hanning,
    noverlap=4096 // 2,
)
#fig.colorbar(im)
"""
ax.set_xlabel("Time (sec)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("Spectrogram of Audio Recording")
ax.set_ylim(0, 6000);
"""

import numpy as np

def ecdf(data):
    """Returns (x) the sorted data and (y) the empirical cumulative-proportion
    of each datum.
    
    Parameters
    ----------
    data : numpy.ndarray, size-N
    
    Returns
    -------
    Tuple[numpy.ndarray shape-(N,), numpy.ndarray shape-(N,)]
        Sorted data, empirical CDF values"""
    data = np.asarray(data).ravel()  # flattens the data
    y = np.linspace(1 / len(data), 1, len(data))  # stores the cumulative proportion associated with each sorted datum
    x = np.sort(data)
    return x, y

"""
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))


ax1.hist(np.log(S).ravel(), bins=100, density=True)
ax1.set_xlabel(r"$\log(|a_{k}|)$")
ax1.set_ylabel(r"Proportion")
ax1.grid()

x, y = ecdf(np.log(S))
ax2.plot(x, y)

ax2.set_xlabel(r"$\log(|a_{k}|)$")
ax2.set_ylabel(r"Cumulative proportion")
ax2.set_title("Cumulative distribution of log-amplitudes")
ax2.grid(True)
"""

log_S = np.log(S).ravel()  # ravel flattens 2D spectrogram into a 1D array
ind = round(len(log_S) * 0.6)  # find the index associated with the 60th percentile log-amplitude
cutoff_log_amplitude = np.partition(log_S, ind)[ind]  # find the actual 60th percentile log-amplitude
#cutoff_log_amplitude

def fanout_pairs(peak_locations, fanout: int = 15, max_dt: int = None):
    """Yields (key, t_m) for every fanout pair.
    Converts to dict later."""
    peaks = sorted(peak_locations, key=lambda p: (p[0], p[1]))
    for m in range(len(peaks)):
        f_m, t_m = peaks[m]
        for n in range(m + 1, min(m + 1 + fanout, len(peaks))):
            f_n, t_n = peaks[n]
            dt = t_n - t_m
            if dt < 0:
                continue
            if max_dt is not None and dt > max_dt:
                continue
            yield (f_m, f_n, dt), t_m

def peaks_to_fingerprint(peak_locations, fanout=15, max_dt=None):
    return dict(fanout_pairs(peak_locations, fanout=fanout, max_dt=max_dt))

def fingerprint_recording(
    data_2d: np.ndarray,
    percentile: float = 75,
    fanout: int = 15,
    max_dt: int = None,
) -> Dict[Tuple[int, int, int], int]:
    """
    End-to-end: log-amplitude spectrogram -> local peaks -> fingerprint.
    """
    amp_min = np.percentile(data_2d.flatten(), percentile)
    neighborhood = generate_binary_structure(rank=2, connectivity=2)
    peaks = local_peak_locations(data_2d, neighborhood, amp_min=amp_min)
    return peaks_to_fingerprint(peaks, fanout=fanout, max_dt=max_dt)

S, freqs, times = mlab.specgram(
    recorded_audio, NFFT=4096, Fs=sampling_rate,
    window=mlab.window_hanning, noverlap=int(4096/2), mode='magnitude',
)

fp = fingerprint_recording(np.log(S + 1e-12), percentile=75, fanout=15)

def get_spectrogram(samples: np.ndarray, sampling_rate: int) -> np.ndarray:
    S, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2),
        mode="magnitude",
    )
    return S

#database here
from pathlib import Path
import librosa
import numpy as np
import matplotlib.mlab as mlab
from scipy.ndimage import generate_binary_structure
from pipeline import local_peak_locations
from pipeline import fanout_pairs

def get_spectrogram(samples: np.ndarray, sampling_rate: int) -> np.ndarray:
    S, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=sampling_rate,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2),
        mode="magnitude",
    )
    return S

def build_song_database(
    songs_folder: str,
    fanout: int = 15,
    percentile: float = 75,
    max_dt: int = None,
) -> Tuple[Dict[DatabaseKey, List[DatabaseEntry]], Dict[int, str]]:
    """
    Returns
    -------
    database : Dict[(f_i, f_j, dt), List[(song_id, t_i)]]
    song_index : Dict[song_id, song_name]
        Lets you map a matched song_id back to its name.
    """
    songs_folder = Path(songs_folder) #songs_folder should be here
    database: Dict[DatabaseKey, List[DatabaseEntry]] = {}
    song_index: Dict[int, str] = {}
    mp3_files = sorted(songs_folder.glob("*.mp3"))

    for song_id, path in enumerate(mp3_files):
        song_index[song_id] = path.stem  # e.g. "Malcom_Todd_Earrings"
        #samples, sr = librosa.load(path, sr=44100, mono=True)

        try:
            samples, sr = librosa.load(path, sr=44100, mono=True)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue
        
        log_S = np.log(get_spectrogram(samples, sr) + 1e-12)

        amp_min = np.percentile(log_S.flatten(), percentile)
        neighborhood = generate_binary_structure(rank=2, connectivity=2)
        peaks = local_peak_locations(log_S, neighborhood, amp_min=amp_min)

        for key, t_m in fanout_pairs(peaks, fanout=fanout, max_dt=max_dt):
            database.setdefault(key, []).append((song_id, t_m))

        #print(f"[{song_id}] {path.name}: {len(peaks)} peaks")

    return database, song_index

#DATABASE HERE
database, song_index = build_song_database("Songs", fanout=15, percentile=75)

#print(song_index)
# {0: 'Malcom_Todd_Earrings', 1: '...', ...}

from microphone import record_audio
from microphone.config import settings
settings.channels = 1

def record_and_get_spectrogram(duration: float):
    p = pyaudio.PyAudio()
    device = p.get_device_info_by_index(0)
    p.terminate()

    frames, sr = record_audio(duration, device=device)
    samples = np.hstack([np.frombuffer(i, np.int16) for i in frames])

    S, freqs, times = mlab.specgram(
        samples,
        NFFT=4096,
        Fs=44100,
        window=mlab.window_hanning,
        noverlap=int(4096 / 2),
        mode="magnitude",
    )
    return S

from collections import Counter

def match_fingerprint(recording_fp, database, song_index):
    votes = Counter()
    for key, t_record in recording_fp.items():
        # hash not in database
        if key not in database:
            continue
        # every occurrence of this hash
        for song_id, t_song in database[key]:
            # align the songs by their starting point
            offset = t_song - t_record
            votes[(song_id, offset)] += 1
    if not votes:
        return None
    (best_song, best_offset), score = votes.most_common(1)[0]

    # Combine votes from different offsets for the same song
    song_votes = Counter()

    for (song_id, offset), score in votes.items():
        song_votes[song_id] += score

    # Get the top two songs
    top = song_votes.most_common(2)

    best_song_id, best_score = top[0]

    if len(top) > 1:
        second_song_id, second_score = top[1]
    else:
        second_song_id = None
        second_score = 0

    # Confidence
    confidence = best_score / (best_score + second_score)
    confidence = str(round(confidence * 100)) + "%"

    # Format best song
    original_song_name = song_index[best_song_id]
    best_song = song_index[best_song_id].replace("_", " ")
    best_song = best_song.split("--")
    best_song = f"{best_song[1]} by {best_song[0]}"

    # Format runner-up
    if second_song_id is not None:
        next_best_match = song_index[second_song_id].replace("_", " ")
        next_best_match = next_best_match.split("--")
        next_best_match = f"{next_best_match[1]} by {next_best_match[0]}"
    else:
        next_best_match = "None"
    
    return {
        "song": best_song,
        "song_file": original_song_name,
        "confidence": confidence,
        "next_best_match": next_best_match,

    }

def identify_song():
    """
    Run end-to-end: record -> spectrogram -> peaks -> fingerprint -> matching -> output.
    """
    spectrogram = record_and_get_spectrogram(5.0)
    spectrogram = np.log(spectrogram + 1e-12)
    cutoff = np.percentile(spectrogram, 75)

    # visualization only
    mask = local_peaks_mask(spectrogram, cutoff)
    # actual peak extraction
    neighborhood = generate_binary_structure(2, 2)

    peak_locations = local_peak_locations(
        spectrogram,
        neighborhood,
        amp_min=cutoff,
    )

    fingerprints = peaks_to_fingerprint(
        peak_locations,
        fanout=15,
        max_dt=None,
    )

    result = match_fingerprint(
        fingerprints,
        database,
        song_index,
    )

    if result is None:
        return None
    else:
        result = match_fingerprint(
            fingerprints,
            database,
            song_index,
        )

        return result



"""
        print((f"Matched song : {result['song']}"))
        return(f"Matched song : {result['song']}")

        print(f"Confidence score: {result['confidence score']}")
        return(f"Confidence score: {result['confidence score']}")

        print(f"Next best match: {result['next best match']}")
        return(f"Next best match: {result['next best match']}")
"""

    


