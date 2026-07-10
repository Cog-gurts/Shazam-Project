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

        print(f"[{song_id}] {path.name}: {len(peaks)} peaks")

    return database, song_index

#DATABASE HERE
database, song_index = build_song_database("Songs", fanout=15, percentile=75)

print(song_index)
# {0: 'Malcom_Todd_Earrings', 1: '...', ...}