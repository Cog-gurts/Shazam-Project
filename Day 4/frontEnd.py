#front end
import streamlit as st
import time
from pathlib import Path
import matplotlib.pyplot as plt
from pipeline import identify_song
from scipy import signal
#frequencies, times, spectrogram_matrix = signal.spectrogram(audio_data, sample_rate)

import streamlit as st

st.title("Sha*zam")
st.caption("Identify any song in 5 seconds.")

if "result" not in st.session_state:
    st.session_state.result = None

if st.button("Tap to Sha*zam!"):

    # Clear previous result immediately
    st.session_state.result = None

    with st.spinner("Listening for 5 seconds..."):
        st.session_state.result = identify_song()

# Only display the current result
if st.session_state.result is not None:
    result = st.session_state.result

    st.success(result["song"])
    cover = Path("AlbumCovers") / f"{result['song_file']}.jpeg"
    if cover.exists():
        st.image(str(cover), width=300)

    st.metric("Confidence", result["confidence"])
    st.write("Next best match:", result["next_best_match"])


