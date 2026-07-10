import pickle
from pathlib import Path

# Note for later: Double check code for path checking

class AudioDatabase:
    """
    Database implementation. Stores song metadata and fingerprints in a dictionary.
    The dictionary is serialized to disk using pickle.

    hash_map = fingerprints
    songs = metadata

    Example in use:
    # initialize
    db = AudioDatabase("data/db.pkl")
    
    # manually add data (note: the a3f9c1 thing is just an example of a footprint)
    db.songs["track1"] = {"title": "24K Magic", "artist": "Bruno Mars"}
    db.hash_map["a3f9c1"] = [("track1", 42)]

    # save it to disk
    db.save()

    # 4. inspect data
    print(db.list_songs())

    
    """

    def __init__(self, path="data/db.pkl"):
        self.path = Path(path)
        self.songs = {}     # song_id -> {"title": ..., "artist": ...}
                                # e.g. {"track1": {"title": "24K Magic", "artist": "Bruno Mars"}}
        self.hash_map = {}  # hash -> [(song_id, offset), ...]
                                # e.g. {"a3f9c1...": [("track1", 42), ("track2", 187)]}
        
        # If file already exists, load it
        if self.path.exists():
            self.load()

    def save(self):
        """
        Saves the database to disk.
        Serialized representation. Would require unpickling to read.
        """
        
        try:
            # If parent dir missing, create it automatically
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                pickle.dump({"songs": self.songs, "hash_map": self.hash_map}, f)
        except:
            print(f"Error saving database to {self.path}")

    def load(self):
        """
        Unpickles and loads files back from disk.
        """

        # Avoid crash if file doesn't exist
        if not self.path.exists():
            raise FileNotFoundError(f"Database file not found at {self.path}.")

        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
            
            self.songs = data["songs"]
            self.hash_map = data["hash_map"]
        except (pickle.UnpicklingError, EOFError):
            print(f"Database file at {self.path} is empty or corrupted.")
            raise

    def list_songs(self):
        return self.songs
    
    def add_song(self, song_id, title, artist):
        """
        Adds a song to the database metadata.
        """
        self.songs[song_id] = {
            "title": title,
            "artist": artist
        }

    def add_hash(self, hash_value, song_id, offset):
        """
        Adds one fingerprint/hash entry to the database.
        """
        if hash_value not in self.hash_map:
            self.hash_map[hash_value] = []

        self.hash_map[hash_value].append((song_id, offset))

    def add_song_full(self, song_id, title, artist, hash_value, offset):
        """
        Adds a song and one fingerprint/hash entry.
        """
        self.add_song(song_id, title, artist)
        self.add_hash(hash_value, song_id, offset)