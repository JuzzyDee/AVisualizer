#!/usr/bin/env python3
"""
Audio Visualizer for the Deaf/Hard of Hearing
Generates comprehensive visual representations of audio files.

This script creates multiple visualization types to help someone
who cannot hear experience music visually.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import librosa
import librosa.display
import argparse
import sys
from pathlib import Path

# Configuration
FIGURE_DPI = 150
COLORMAP_MAIN = 'magma'
COLORMAP_DIVERGING = 'coolwarm'


def load_audio(filepath):
    """Load audio file and return time series and sample rate."""
    print(f"Loading audio file: {filepath}")
    y, sr = librosa.load(filepath, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Samples: {len(y):,}")
    return y, sr


def create_output_dir(base_path):
    """Create output directory for visualizations."""
    output_dir = base_path.parent / f"{base_path.stem}_visualizations"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir


def save_figure(fig, output_dir, name, tight=True):
    """Save figure to output directory."""
    filepath = output_dir / f"{name}.png"
    if tight:
        fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    else:
        fig.savefig(filepath, dpi=FIGURE_DPI, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {name}.png")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_waveform(y, sr, output_dir):
    """
    1. WAVEFORM - Basic amplitude over time
    Shows the raw audio signal - peaks indicate loud moments,
    flat areas indicate quiet moments.
    """
    print("Generating: Waveform...")
    fig, ax = plt.subplots(figsize=(16, 4))

    times = np.linspace(0, len(y)/sr, len(y))
    ax.plot(times, y, color='#2E86AB', linewidth=0.3, alpha=0.8)
    ax.fill_between(times, y, alpha=0.3, color='#2E86AB')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Waveform - Audio Amplitude Over Time\n(Peaks = Loud, Flat = Quiet)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(y)/sr)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '01_waveform')


def plot_waveform_envelope(y, sr, output_dir):
    """
    2. WAVEFORM ENVELOPE - Smoothed amplitude showing dynamics
    Shows overall loudness changes without the rapid oscillations.
    """
    print("Generating: Waveform Envelope...")
    fig, ax = plt.subplots(figsize=(16, 4))

    # Compute envelope using Hilbert transform
    analytic_signal = signal.hilbert(y)
    envelope = np.abs(analytic_signal)

    # Smooth the envelope
    window_size = int(sr * 0.05)  # 50ms window
    envelope_smooth = gaussian_filter1d(envelope, sigma=window_size)

    times = np.linspace(0, len(y)/sr, len(y))
    ax.fill_between(times, envelope_smooth, alpha=0.7, color='#E94F37')
    ax.plot(times, envelope_smooth, color='#E94F37', linewidth=0.5)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Loudness', fontsize=12)
    ax.set_title('Volume Envelope - Overall Loudness Over Time\n(Higher = Louder sections)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, len(y)/sr)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '02_volume_envelope')


def plot_spectrogram(y, sr, output_dir):
    """
    3. SPECTROGRAM - Frequency content over time
    Shows what pitches/frequencies are playing at each moment.
    Bottom = low/bass notes, Top = high/treble notes.
    Brightness = loudness of that frequency.
    """
    print("Generating: Spectrogram...")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Compute spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log',
                                    ax=ax, cmap=COLORMAP_MAIN)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Hz) - Low to High pitch', fontsize=12)
    ax.set_title('Spectrogram - All Frequencies Over Time\n(Bottom = Bass/Low, Top = Treble/High, Bright = Loud)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Loudness (dB)', fontsize=11)

    save_figure(fig, output_dir, '03_spectrogram')


def plot_mel_spectrogram(y, sr, output_dir):
    """
    4. MEL SPECTROGRAM - Human-perception-weighted frequency view
    Similar to spectrogram but scaled to match how humans perceive pitch.
    """
    print("Generating: Mel Spectrogram...")
    fig, ax = plt.subplots(figsize=(16, 8))

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr/2)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                                    ax=ax, cmap=COLORMAP_MAIN, fmax=sr/2)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (Mel scale) - Perceived pitch', fontsize=12)
    ax.set_title('Mel Spectrogram - Frequencies Scaled to Human Pitch Perception\n(How we naturally hear pitch differences)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.set_label('Loudness (dB)', fontsize=11)

    save_figure(fig, output_dir, '04_mel_spectrogram')


def plot_chromagram(y, sr, output_dir):
    """
    5. CHROMAGRAM - Musical notes/chords over time
    Shows the 12 musical notes (C, C#, D, etc.) and their intensity.
    Great for seeing chord progressions and melody.
    """
    print("Generating: Chromagram...")
    fig, ax = plt.subplots(figsize=(16, 6))

    # Compute chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma',
                                    ax=ax, cmap='YlOrRd')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Musical Note', fontsize=12)
    ax.set_title('Chromagram - Musical Notes Over Time\n(Shows which of the 12 notes are playing - chord progressions)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Note Intensity', fontsize=11)

    save_figure(fig, output_dir, '05_chromagram')


def plot_tonnetz(y, sr, output_dir):
    """
    6. TONNETZ - Harmonic relationships
    Shows tonal/harmonic content using music theory relationships.
    """
    print("Generating: Tonnetz (Harmonic Space)...")
    fig, ax = plt.subplots(figsize=(16, 6))

    # Compute tonnetz
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)

    img = librosa.display.specshow(tonnetz, sr=sr, x_axis='time',
                                    ax=ax, cmap=COLORMAP_DIVERGING)

    ax.set_ylabel('Tonal Dimension', fontsize=12)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Tonnetz - Harmonic/Tonal Relationships\n(Shows musical harmony and chord relationships)',
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(6))
    ax.set_yticklabels(['Fifth (x)', 'Fifth (y)', 'Minor (x)',
                        'Minor (y)', 'Major (x)', 'Major (y)'])

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Intensity', fontsize=11)

    save_figure(fig, output_dir, '06_tonnetz')


def plot_spectral_centroid(y, sr, output_dir):
    """
    7. SPECTRAL CENTROID - Brightness of sound over time
    Higher values = brighter/sharper sound, Lower = darker/duller sound.
    """
    print("Generating: Spectral Centroid (Brightness)...")
    fig, ax = plt.subplots(figsize=(16, 5))

    # Compute spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    frames = range(len(cent))
    times = librosa.frames_to_time(frames, sr=sr)

    # Normalize for coloring
    cent_norm = (cent - cent.min()) / (cent.max() - cent.min())

    # Create colored line segments
    points = np.array([times, cent]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = plt.Normalize(cent.min(), cent.max())
    lc = LineCollection(segments, cmap='plasma', norm=norm)
    lc.set_array(cent)
    lc.set_linewidth(2)

    line = ax.add_collection(lc)
    ax.set_xlim(times.min(), times.max())
    ax.set_ylim(cent.min() * 0.9, cent.max() * 1.1)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Spectral Centroid (Hz)', fontsize=12)
    ax.set_title('Spectral Centroid - Sound Brightness Over Time\n(High = Bright/Sharp sound, Low = Dark/Dull sound)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label('Brightness (Hz)', fontsize=11)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '07_spectral_centroid')


def plot_spectral_bandwidth(y, sr, output_dir):
    """
    8. SPECTRAL BANDWIDTH - How spread out the frequencies are
    Wide = rich/complex sound, Narrow = pure/simple sound.
    """
    print("Generating: Spectral Bandwidth...")
    fig, ax = plt.subplots(figsize=(16, 5))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frames = range(len(spec_bw))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, spec_bw, alpha=0.6, color='#7B2CBF')
    ax.plot(times, spec_bw, color='#7B2CBF', linewidth=1)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Bandwidth (Hz)', fontsize=12)
    ax.set_title('Spectral Bandwidth - Sound Richness/Complexity\n(Wide = Rich/Complex, Narrow = Pure/Simple)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '08_spectral_bandwidth')


def plot_spectral_rolloff(y, sr, output_dir):
    """
    9. SPECTRAL ROLLOFF - Where most of the energy is concentrated
    Shows the frequency below which 85% of the sound energy exists.
    """
    print("Generating: Spectral Rolloff...")
    fig, ax = plt.subplots(figsize=(16, 5))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    frames = range(len(rolloff))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, rolloff, alpha=0.6, color='#00A896')
    ax.plot(times, rolloff, color='#00A896', linewidth=1)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Rolloff Frequency (Hz)', fontsize=12)
    ax.set_title('Spectral Rolloff - Where 85% of Sound Energy Lives\n(Higher = More high-frequency content)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '09_spectral_rolloff')


def plot_rms_energy(y, sr, output_dir):
    """
    10. RMS ENERGY - Overall loudness/power over time
    Shows the intensity and dynamics of the music.
    """
    print("Generating: RMS Energy (Loudness)...")
    fig, ax = plt.subplots(figsize=(16, 4))

    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, rms, alpha=0.7, color='#F77F00')
    ax.plot(times, rms, color='#D62828', linewidth=1)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Energy (RMS)', fontsize=12)
    ax.set_title('RMS Energy - Overall Loudness/Power Over Time\n(Peaks = Intense moments, Valleys = Quieter sections)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '10_rms_energy')


def plot_zero_crossing_rate(y, sr, output_dir):
    """
    11. ZERO CROSSING RATE - Texture/noisiness indicator
    High values indicate noisy/percussive sounds, low = tonal/smooth sounds.
    """
    print("Generating: Zero Crossing Rate (Texture)...")
    fig, ax = plt.subplots(figsize=(16, 4))

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    frames = range(len(zcr))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, zcr, alpha=0.6, color='#84A98C')
    ax.plot(times, zcr, color='#2D6A4F', linewidth=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Zero Crossing Rate', fontsize=12)
    ax.set_title('Zero Crossing Rate - Sound Texture\n(High = Noisy/Percussive, Low = Smooth/Tonal)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '11_zero_crossing_rate')


def plot_onset_strength(y, sr, output_dir):
    """
    12. ONSET STRENGTH - Where new sounds/notes begin
    Peaks indicate the start of new notes, beats, or events.
    """
    print("Generating: Onset Strength (Note Attacks)...")
    fig, ax = plt.subplots(figsize=(16, 4))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = range(len(onset_env))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, onset_env, alpha=0.6, color='#FF006E')
    ax.plot(times, onset_env, color='#FF006E', linewidth=0.8)

    # Mark detected onsets
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    ax.vlines(onsets, 0, onset_env.max(), color='#3A0CA3', alpha=0.5,
              linewidth=0.5, label='Detected note starts')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Onset Strength', fontsize=12)
    ax.set_title('Onset Strength - New Notes/Sounds Starting\n(Peaks and lines = New notes or beats beginning)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '12_onset_strength')


def plot_beat_track(y, sr, output_dir):
    """
    13. BEAT TRACKING - The rhythm/pulse of the music
    Shows where the beats are and the tempo structure.
    """
    print("Generating: Beat Track (Rhythm)...")
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    # Compute tempo and beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)

    # Handle tempo - it might be an array
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_val = float(tempo)

    # Top plot: waveform with beat markers
    times = np.linspace(0, len(y)/sr, len(y))
    axes[0].plot(times, y, color='#2E86AB', linewidth=0.3, alpha=0.6)
    axes[0].vlines(beat_times, -1, 1, color='#D62828', alpha=0.8,
                   linewidth=1, label='Beats')
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title(f'Beat Tracking - Detected Tempo: {tempo_val:.1f} BPM\n(Red lines = Beat positions)',
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(0, len(y)/sr)

    # Bottom plot: onset strength with beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = range(len(onset_env))
    otimes = librosa.frames_to_time(frames, sr=sr)

    axes[1].fill_between(otimes, onset_env, alpha=0.5, color='#F77F00')
    axes[1].vlines(beat_times, 0, onset_env.max(), color='#D62828',
                   alpha=0.8, linewidth=1)
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_ylabel('Onset Strength', fontsize=11)
    axes[1].set_xlim(0, len(y)/sr)

    plt.tight_layout()
    save_figure(fig, output_dir, '13_beat_tracking', tight=False)


def plot_tempogram(y, sr, output_dir):
    """
    14. TEMPOGRAM - Tempo/rhythm patterns over time
    Shows how the rhythm structure changes throughout the piece.
    """
    print("Generating: Tempogram (Rhythm Patterns)...")
    fig, ax = plt.subplots(figsize=(16, 6))

    # Compute tempogram
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

    img = librosa.display.specshow(tempogram, sr=sr, x_axis='time',
                                    y_axis='tempo', ax=ax, cmap='magma')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Tempo (BPM)', fontsize=12)
    ax.set_title('Tempogram - Rhythm/Tempo Patterns Over Time\n(Bright horizontal bands = Strong rhythmic patterns)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Strength', fontsize=11)

    save_figure(fig, output_dir, '14_tempogram')


def plot_mfcc(y, sr, output_dir):
    """
    15. MFCCs - Timbral texture (sound color/character)
    Shows the "color" or character of the sound - what makes
    a piano sound different from a guitar.
    """
    print("Generating: MFCCs (Sound Character/Timbre)...")
    fig, ax = plt.subplots(figsize=(16, 6))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax,
                                    cmap=COLORMAP_DIVERGING)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('MFCC Coefficient', fontsize=12)
    ax.set_title('MFCCs - Sound Character/Timbre ("Color" of the sound)\n(Different patterns = Different instrument sounds)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Coefficient Value', fontsize=11)

    save_figure(fig, output_dir, '15_mfcc')


def plot_spectral_contrast(y, sr, output_dir):
    """
    16. SPECTRAL CONTRAST - Difference between peaks and valleys
    Shows the difference between loud and quiet frequency bands.
    """
    print("Generating: Spectral Contrast...")
    fig, ax = plt.subplots(figsize=(16, 6))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    img = librosa.display.specshow(contrast, sr=sr, x_axis='time', ax=ax,
                                    cmap='PRGn')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency Band', fontsize=12)
    ax.set_title('Spectral Contrast - Dynamic Range per Frequency Band\n(High contrast = Clear/distinct sounds, Low = Muddy/blended)',
                 fontsize=14, fontweight='bold')

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label('Contrast (dB)', fontsize=11)

    save_figure(fig, output_dir, '16_spectral_contrast')


def plot_harmonic_percussive(y, sr, output_dir):
    """
    17. HARMONIC vs PERCUSSIVE separation
    Separates sustained sounds (instruments, vocals) from
    sharp attack sounds (drums, percussion).
    """
    print("Generating: Harmonic vs Percussive Separation...")

    # Separate harmonic and percussive components
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    times = np.linspace(0, len(y)/sr, len(y))

    # Original
    axes[0].plot(times, y, color='#2E86AB', linewidth=0.3)
    axes[0].set_ylabel('Original', fontsize=11)
    axes[0].set_title('Harmonic vs Percussive Separation\n(Splitting sustained notes from drum hits/attacks)',
                     fontsize=14, fontweight='bold')

    # Harmonic (sustained notes, melody, chords)
    axes[1].plot(times, y_harmonic, color='#06D6A0', linewidth=0.3)
    axes[1].set_ylabel('Harmonic\n(Melody/Chords)', fontsize=11)

    # Percussive (drums, attacks)
    axes[2].plot(times, y_percussive, color='#EF476F', linewidth=0.3)
    axes[2].set_ylabel('Percussive\n(Drums/Attacks)', fontsize=11)
    axes[2].set_xlabel('Time (seconds)', fontsize=12)

    for ax in axes:
        ax.set_xlim(0, len(y)/sr)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_dir, '17_harmonic_percussive', tight=False)


def plot_frequency_bands(y, sr, output_dir):
    """
    18. FREQUENCY BANDS - Energy in bass, mid, and treble
    Shows the balance of low, mid, and high frequencies over time.
    """
    print("Generating: Frequency Bands (Bass/Mid/Treble)...")

    # Compute spectrogram
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    # Define frequency bands
    bands = {
        'Sub-bass (20-60 Hz)': (20, 60),
        'Bass (60-250 Hz)': (60, 250),
        'Low-mid (250-500 Hz)': (250, 500),
        'Mid (500-2000 Hz)': (500, 2000),
        'High-mid (2000-4000 Hz)': (2000, 4000),
        'Treble (4000-20000 Hz)': (4000, 20000)
    }

    colors = ['#540B0E', '#9E2A2B', '#E09F3E', '#FFF3B0', '#335C67', '#2E86AB']

    fig, ax = plt.subplots(figsize=(16, 6))

    times = librosa.frames_to_time(range(S.shape[1]), sr=sr)

    band_energies = []
    for (name, (low, high)), color in zip(bands.items(), colors):
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() > 0:
            energy = S[mask].mean(axis=0)
            energy_smooth = gaussian_filter1d(energy, sigma=5)
            band_energies.append((name, energy_smooth, color))

    # Stack plot
    energies = np.array([e[1] for e in band_energies])
    energies_norm = energies / energies.sum(axis=0, keepdims=True)

    ax.stackplot(times, energies_norm, labels=[e[0] for e in band_energies],
                 colors=colors, alpha=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Relative Energy', fontsize=12)
    ax.set_title('Frequency Band Distribution Over Time\n(Shows balance of bass, mids, and treble)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_dir, '18_frequency_bands')


def plot_dynamic_range(y, sr, output_dir):
    """
    19. DYNAMIC RANGE - Loud vs quiet sections highlighted
    Shows the contrast between loud and quiet parts.
    """
    print("Generating: Dynamic Range...")
    fig, ax = plt.subplots(figsize=(16, 5))

    # Compute RMS in frames
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)

    # Normalize RMS to 0-1 range
    rms_norm = (rms - rms.min()) / (rms.max() - rms.min())

    # Create gradient fill based on loudness
    for i in range(len(times) - 1):
        color = plt.cm.RdYlGn_r(rms_norm[i])
        ax.fill_between([times[i], times[i+1]], [0, 0], [rms[i], rms[i+1]],
                        color=color, alpha=0.8)

    ax.plot(times, rms, color='black', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Loudness', fontsize=12)
    ax.set_title('Dynamic Range - Volume Variation\n(Red = Loud peaks, Green = Quiet sections)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.set_ylim(0, None)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Relative Loudness', fontsize=11)

    save_figure(fig, output_dir, '19_dynamic_range')


def plot_spectral_flatness(y, sr, output_dir):
    """
    20. SPECTRAL FLATNESS - Noise vs tonal content
    High = noise-like (drums, percussion), Low = tonal (melody, chords).
    """
    print("Generating: Spectral Flatness (Noise vs Tone)...")
    fig, ax = plt.subplots(figsize=(16, 4))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    frames = range(len(flatness))
    times = librosa.frames_to_time(frames, sr=sr)

    ax.fill_between(times, flatness, alpha=0.6, color='#9B5DE5')
    ax.plot(times, flatness, color='#9B5DE5', linewidth=0.8)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Spectral Flatness', fontsize=12)
    ax.set_title('Spectral Flatness - Noise vs Tonal Content\n(High = Noisy/percussive, Low = Tonal/melodic)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, times.max())
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_dir, '20_spectral_flatness')


def plot_combined_dashboard(y, sr, output_dir, base_path):
    """
    21. COMBINED DASHBOARD - All key visualizations in one view
    A comprehensive overview combining multiple visualizations.
    """
    print("Generating: Combined Dashboard...")
    fig = plt.figure(figsize=(20, 16))

    # Create grid
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.2)

    # 1. Mel Spectrogram (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    S_db = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',
                              ax=ax1, cmap=COLORMAP_MAIN)
    ax1.set_title('Mel Spectrogram (Pitch Content)', fontweight='bold')

    # 2. Chromagram (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma',
                              ax=ax2, cmap='YlOrRd')
    ax2.set_title('Chromagram (Musical Notes)', fontweight='bold')

    # 3. RMS Energy (second row left)
    ax3 = fig.add_subplot(gs[1, 0])
    rms = librosa.feature.rms(y=y)[0]
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr)
    ax3.fill_between(times, rms, alpha=0.7, color='#F77F00')
    ax3.set_xlim(0, times.max())
    ax3.set_title('Volume/Energy Over Time', fontweight='bold')
    ax3.set_xlabel('Time (s)')

    # 4. Spectral Centroid (second row right)
    ax4 = fig.add_subplot(gs[1, 1])
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times_c = librosa.frames_to_time(range(len(cent)), sr=sr)
    ax4.fill_between(times_c, cent, alpha=0.6, color='#9B5DE5')
    ax4.set_xlim(0, times_c.max())
    ax4.set_title('Brightness Over Time', fontweight='bold')
    ax4.set_xlabel('Time (s)')

    # 5. Onset Strength with Beats (third row, full width)
    ax5 = fig.add_subplot(gs[2, :])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times_o = librosa.frames_to_time(range(len(onset_env)), sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    ax5.fill_between(times_o, onset_env, alpha=0.5, color='#FF006E')
    ax5.vlines(beat_times, 0, onset_env.max(), color='#3A0CA3', alpha=0.6, linewidth=1)
    ax5.set_xlim(0, times_o.max())
    ax5.set_title('Rhythm & Beats (vertical lines = beat positions)', fontweight='bold')
    ax5.set_xlabel('Time (s)')

    # 6. Frequency Bands (bottom, full width)
    ax6 = fig.add_subplot(gs[3, :])
    S_full = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    bands = [(60, 250), (250, 2000), (2000, 20000)]
    band_names = ['Bass', 'Mid', 'Treble']
    colors = ['#E63946', '#F4A261', '#2A9D8F']

    times_f = librosa.frames_to_time(range(S_full.shape[1]), sr=sr)
    for (low, high), name, color in zip(bands, band_names, colors):
        mask = (freqs >= low) & (freqs < high)
        if mask.sum() > 0:
            energy = gaussian_filter1d(S_full[mask].mean(axis=0), sigma=5)
            ax6.plot(times_f, energy / energy.max(), label=name, color=color, linewidth=1.5)

    ax6.set_xlim(0, times_f.max())
    ax6.set_title('Frequency Band Balance (Bass/Mid/Treble)', fontweight='bold')
    ax6.set_xlabel('Time (s)')
    ax6.legend(loc='upper right')

    plt.suptitle(f'Audio Visualization Dashboard\n"{base_path.stem}"',
                 fontsize=16, fontweight='bold', y=0.98)

    save_figure(fig, output_dir, '21_combined_dashboard', tight=False)


def plot_3d_spectrogram(y, sr, output_dir):
    """
    22. 3D SPECTROGRAM - Frequency, time, and amplitude in 3D
    A three-dimensional view of the audio.
    """
    print("Generating: 3D Spectrogram...")
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Compute spectrogram with reduced resolution for 3D
    hop_length = 2048
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    # Downsample for visualization
    step_t = max(1, S_db.shape[1] // 200)
    step_f = max(1, S_db.shape[0] // 100)
    S_down = S_db[::step_f, ::step_t]

    # Create meshgrid
    times = librosa.frames_to_time(range(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr)

    times_down = times[::step_t]
    freqs_down = freqs[::step_f]

    T, F = np.meshgrid(times_down, freqs_down)

    # Plot surface
    surf = ax.plot_surface(T, F, S_down, cmap='magma',
                           linewidth=0, antialiased=True, alpha=0.9)

    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_zlabel('Amplitude (dB)', fontsize=11)
    ax.set_title('3D Spectrogram\n(Time × Frequency × Loudness)',
                 fontsize=14, fontweight='bold')

    ax.view_init(elev=30, azim=45)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Amplitude (dB)')

    save_figure(fig, output_dir, '22_3d_spectrogram')


def create_visualization_guide(output_dir, duration, tempo, title):
    """Create a text guide explaining all visualizations."""
    guide_path = output_dir / "VISUALIZATION_GUIDE.txt"

    # Handle tempo - it might be an array
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_val = float(tempo)

    guide_text = f"""
================================================================================
    AUDIO VISUALIZATION GUIDE
    "{title}"
================================================================================

Duration: {duration:.2f} seconds
Detected Tempo: {tempo_val:.1f} BPM

This folder contains {22} different visual representations of the audio file.
Each visualization shows a different aspect of how the music sounds.

--------------------------------------------------------------------------------
BASIC VISUALIZATIONS (Start Here)
--------------------------------------------------------------------------------

01_waveform.png
   What it shows: The raw audio signal over time
   How to read it: Tall peaks = loud moments, flat areas = quiet moments
   The shape shows the overall "texture" of the sound

02_volume_envelope.png
   What it shows: Overall loudness smoothed out over time
   How to read it: Higher = louder, watch for crescendos (getting louder)
   and decrescendos (getting quieter)

--------------------------------------------------------------------------------
FREQUENCY/PITCH VISUALIZATIONS
--------------------------------------------------------------------------------

03_spectrogram.png
   What it shows: ALL frequencies (pitches) over time
   How to read it:
   - Bottom = low/bass notes, Top = high/treble notes
   - Brighter colors = louder at that frequency
   - Horizontal lines = sustained notes
   - Vertical patterns = rhythmic hits/drums

04_mel_spectrogram.png
   What it shows: Same as spectrogram but scaled to human hearing
   How to read it: Spacing matches how we perceive pitch differences
   Low notes spread apart, high notes compressed (like piano keys)

05_chromagram.png
   What it shows: The 12 musical notes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
   How to read it: Bright horizontal bands = that note is playing
   Watch for patterns - these are chord progressions!

06_tonnetz.png
   What it shows: Musical harmony relationships
   How to read it: Shows how notes relate to each other harmonically
   Patterns indicate chord types and key changes

--------------------------------------------------------------------------------
SOUND CHARACTER VISUALIZATIONS
--------------------------------------------------------------------------------

07_spectral_centroid.png
   What it shows: "Brightness" of the sound
   How to read it: High values = bright/sharp sound (like cymbal)
   Low values = dark/mellow sound (like bass)

08_spectral_bandwidth.png
   What it shows: How "spread out" the frequencies are
   How to read it: Wide = rich, complex sound (orchestra)
   Narrow = pure, simple sound (flute solo)

09_spectral_rolloff.png
   What it shows: Where most of the sound energy is concentrated
   How to read it: Higher = more high-frequency content

15_mfcc.png
   What it shows: The "character" or "color" of the sound (timbre)
   How to read it: Different patterns = different instrument sounds
   This is what makes a piano sound different from a trumpet

16_spectral_contrast.png
   What it shows: Difference between loud and quiet frequency bands
   How to read it: High contrast = clear, distinct sounds
   Low contrast = muddy, blended sounds

--------------------------------------------------------------------------------
RHYTHM & DYNAMICS VISUALIZATIONS
--------------------------------------------------------------------------------

10_rms_energy.png
   What it shows: Overall power/intensity over time
   How to read it: Peaks = intense/powerful moments
   Valleys = calmer sections

12_onset_strength.png
   What it shows: Where new notes/sounds begin
   How to read it: Peaks and vertical lines = new notes starting
   Great for seeing the rhythm and when instruments come in

13_beat_tracking.png
   What it shows: The detected beats/pulse of the music
   How to read it: Red vertical lines = beat positions
   The spacing shows the tempo and rhythm

14_tempogram.png
   What it shows: Rhythm patterns over time
   How to read it: Bright horizontal bands = strong rhythmic patterns
   at that tempo (BPM). Changes = tempo variations

19_dynamic_range.png
   What it shows: Volume variation with color coding
   How to read it: Red = loud peaks, Green = quiet sections
   Shows the dramatic contrast in the music

--------------------------------------------------------------------------------
TEXTURE VISUALIZATIONS
--------------------------------------------------------------------------------

11_zero_crossing_rate.png
   What it shows: Sound texture (smooth vs rough)
   How to read it: High = noisy/percussive (drums, cymbals)
   Low = smooth/tonal (sustained notes, vocals)

20_spectral_flatness.png
   What it shows: Noise vs tonal content
   How to read it: High = noise-like (percussion, breath sounds)
   Low = tonal/melodic (notes, chords)

--------------------------------------------------------------------------------
COMPONENT SEPARATION
--------------------------------------------------------------------------------

17_harmonic_percussive.png
   What it shows: The audio split into two parts
   - HARMONIC: Sustained sounds (piano, brass, strings)
   - PERCUSSIVE: Sharp attacks (drums, plucks)
   How to read it: Top = original, Middle = melody/chords, Bottom = drums/hits

18_frequency_bands.png
   What it shows: Balance of bass, mid, and treble over time
   How to read it: The colored areas show which frequencies dominate
   at each moment. Watch how the balance shifts!

--------------------------------------------------------------------------------
OVERVIEW VISUALIZATIONS
--------------------------------------------------------------------------------

21_combined_dashboard.png
   What it shows: Multiple key visualizations in one view
   How to read it: A comprehensive overview of the piece
   Good for getting the overall picture quickly

22_3d_spectrogram.png
   What it shows: Time, frequency, and amplitude in 3D
   How to read it: Peaks = loud frequencies, valleys = quiet
   Gives a "landscape" view of the music

================================================================================
TIPS FOR EXPERIENCING THE MUSIC
================================================================================

1. Start with the Combined Dashboard (21) to get an overview

2. For MELODY and HARMONY: Focus on the Chromagram (05) and
   Mel Spectrogram (04)

3. For RHYTHM: Look at Beat Tracking (13), Onset Strength (12),
   and Tempogram (14)

4. For EMOTIONAL DYNAMICS: Watch the RMS Energy (10) and
   Dynamic Range (19)

5. For TEXTURE and SOUND CHARACTER: Explore MFCC (15) and
   Spectral Centroid (07)

================================================================================
"""

    with open(guide_path, 'w') as f:
        f.write(guide_text)

    print(f"  Saved: VISUALIZATION_GUIDE.txt")


def main():
    """Main function to run all visualizations."""
    global FIGURE_DPI

    parser = argparse.ArgumentParser(
        description='Generate comprehensive visual representations of audio files. '
                    'Creates 22 different visualizations to help experience music visually.'
    )
    parser.add_argument(
        'file',
        help='Path to the audio file (mp3, wav, flac, ogg, etc.)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Custom output directory (default: <filename>_visualizations in the same folder as the audio file)',
        default=None
    )
    parser.add_argument(
        '--dpi',
        help=f'Figure resolution in DPI (default: {FIGURE_DPI})',
        type=int,
        default=None
    )

    args = parser.parse_args()

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"Error: File not found: {audio_path}")
        sys.exit(1)

    if args.dpi is not None:
        FIGURE_DPI = args.dpi

    title = audio_path.stem

    print("=" * 60)
    print("AUDIO VISUALIZER")
    print("Let Claude hear your music")
    print("=" * 60)
    print()

    # Load audio
    y, sr = load_audio(audio_path)
    duration = librosa.get_duration(y=y, sr=sr)

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = create_output_dir(audio_path)

    print()
    print("Generating visualizations...")
    print("-" * 40)

    # Generate all visualizations
    plot_waveform(y, sr, output_dir)
    plot_waveform_envelope(y, sr, output_dir)
    plot_spectrogram(y, sr, output_dir)
    plot_mel_spectrogram(y, sr, output_dir)
    plot_chromagram(y, sr, output_dir)
    plot_tonnetz(y, sr, output_dir)
    plot_spectral_centroid(y, sr, output_dir)
    plot_spectral_bandwidth(y, sr, output_dir)
    plot_spectral_rolloff(y, sr, output_dir)
    plot_rms_energy(y, sr, output_dir)
    plot_zero_crossing_rate(y, sr, output_dir)
    plot_onset_strength(y, sr, output_dir)
    plot_beat_track(y, sr, output_dir)
    plot_tempogram(y, sr, output_dir)
    plot_mfcc(y, sr, output_dir)
    plot_spectral_contrast(y, sr, output_dir)
    plot_harmonic_percussive(y, sr, output_dir)
    plot_frequency_bands(y, sr, output_dir)
    plot_dynamic_range(y, sr, output_dir)
    plot_spectral_flatness(y, sr, output_dir)
    plot_combined_dashboard(y, sr, output_dir, audio_path)
    plot_3d_spectrogram(y, sr, output_dir)

    # Get tempo for guide
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Create guide
    print()
    print("Creating visualization guide...")
    create_visualization_guide(output_dir, duration, tempo, title)

    print()
    print("=" * 60)
    print("COMPLETE!")
    print(f"Generated 22 visualizations + guide in:")
    print(f"  {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
