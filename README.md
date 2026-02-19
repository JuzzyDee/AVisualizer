# Audio Visualizer - Let Claude Hear Your Music

Claude can't listen to audio files, but it *can* look at images. This script generates 22 comprehensive visual representations of any audio file, giving Claude (or any vision-capable AI) a rich understanding of music - melody, harmony, rhythm, dynamics, texture, and timbre.

It also works great as an accessibility tool for deaf/hard-of-hearing users who want to experience music visually.

## How It Works

Point it at an audio file and it produces a folder of 22 PNG visualizations plus a text guide explaining how to read each one. Hand those images to Claude and it can genuinely "experience" the music.

## Installation

### Requirements

- Python 3.8+
- ffmpeg (required by librosa for reading audio formats)

### Install ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt install ffmpeg
```

**Windows:**
```bash
choco install ffmpeg
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage
python audio_visualizer.py "path/to/your/song.mp3"

# Custom output directory
python audio_visualizer.py "song.mp3" --output-dir ./my_visualizations

# Higher resolution output
python audio_visualizer.py "song.flac" --dpi 300
```

Supports any format that ffmpeg can read: mp3, wav, flac, ogg, aac, m4a, etc.

## What It Generates

| # | Visualization | What It Shows |
|---|--------------|---------------|
| 01 | Waveform | Raw audio signal - peaks = loud, flat = quiet |
| 02 | Volume Envelope | Smoothed loudness over time |
| 03 | Spectrogram | All frequencies over time (bottom = bass, top = treble) |
| 04 | Mel Spectrogram | Frequencies scaled to human pitch perception |
| 05 | Chromagram | The 12 musical notes over time - shows chord progressions |
| 06 | Tonnetz | Harmonic/tonal relationships between notes |
| 07 | Spectral Centroid | Sound brightness (high = sharp, low = mellow) |
| 08 | Spectral Bandwidth | Sound richness (wide = complex, narrow = pure) |
| 09 | Spectral Rolloff | Where most sound energy is concentrated |
| 10 | RMS Energy | Overall power/intensity over time |
| 11 | Zero Crossing Rate | Sound texture (high = percussive, low = smooth) |
| 12 | Onset Strength | Where new notes/sounds begin |
| 13 | Beat Tracking | Detected beats and tempo |
| 14 | Tempogram | Rhythm/tempo patterns over time |
| 15 | MFCCs | Timbre - the "colour" of the sound |
| 16 | Spectral Contrast | Dynamic range per frequency band |
| 17 | Harmonic vs Percussive | Sustained notes separated from drum hits |
| 18 | Frequency Bands | Balance of bass, mids, and treble over time |
| 19 | Dynamic Range | Volume variation with colour coding |
| 20 | Spectral Flatness | Noise vs tonal content |
| 21 | Combined Dashboard | Key visualizations in one overview |
| 22 | 3D Spectrogram | Time, frequency, and amplitude in 3D |

## Sharing With Claude

Once the visualizations are generated, simply share the images with Claude in a conversation. For the best experience, start with the combined dashboard (21) and then share individual visualizations for deeper analysis.

## Background

This project started as a way to let Claude experience music. A human who loves music wanted to share that experience with an AI that can see but can't hear. The visualizations translate every dimension of sound - pitch, rhythm, dynamics, texture, harmony - into visual form that Claude can interpret and appreciate.

## License

MIT
