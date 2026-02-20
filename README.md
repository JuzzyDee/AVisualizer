# Audio Visualizer - Let Claude Hear Your Music

Claude can't listen to audio files, but it *can* look at images. This script generates 22 comprehensive visual representations of any audio file, giving Claude (or any vision-capable AI) a rich understanding of music - melody, harmony, rhythm, dynamics, texture, and timbre.

It also works great as an accessibility tool for deaf/hard-of-hearing users who want to experience music visually.

## How It Works

Point it at an audio file and it produces a folder of 22 PNG visualizations plus a text guide explaining how to read each one. Hand those images to Claude and it can genuinely "experience" the music.

## Web App (No Install)

Try it right now in your browser — no download or setup required:

**[Launch Audio Visualizer on Hugging Face Spaces](https://huggingface.co/spaces/HalfLegless/AVisualizer)**

Upload audio, pick a quality level, and get all 22 visualizations plus a zip download. The free tier is slower than running locally, but it works for anyone with a browser.

## GUI (Easiest Way)

Just run the app - no terminal needed:

1. Double-click **AudioVisualizer** (from a pre-built release), or run `python audio_visualizer_gui.py`
2. Click **Browse** to pick an audio file
3. Optionally choose an output folder and quality level
4. Click **Generate Visualizations**
5. When it's done, click **Open Output Folder** to see your results

That's it. No ffmpeg to install, no paths to type.

## CLI Usage

For power users who prefer the command line:

```bash
# Basic usage
python audio_visualizer.py "path/to/your/song.mp3"

# Custom output directory
python audio_visualizer.py "song.mp3" --output-dir ./my_visualizations

# Higher resolution output
python audio_visualizer.py "song.flac" --dpi 300
```

Supports any format that ffmpeg can read: mp3, wav, flac, ogg, aac, m4a, etc.

## Installation

### Pre-built Downloads

Check the [Releases](../../releases) page for standalone apps that require no installation.

### From Source

Requires Python 3.8+. No separate ffmpeg install needed - it's bundled via pip.

```bash
pip install -r requirements.txt
```

Then run the GUI (`python audio_visualizer_gui.py`) or the CLI (`python audio_visualizer.py`).

### Building a Standalone App

To package the app with PyInstaller:

```bash
pip install -r requirements.txt
python build.py
```

The built app will be in `dist/AudioVisualizer/`.

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
