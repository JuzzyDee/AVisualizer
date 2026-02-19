#!/usr/bin/env python3
"""
Audio Visualizer GUI
A simple graphical interface for generating audio visualizations.
No terminal, no paths to type - just click and go.
"""

import matplotlib
matplotlib.use('Agg')  # MUST be before any pyplot/librosa import

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import shutil
import sys
import gc
import time
import platform
from pathlib import Path

import matplotlib.pyplot as plt

import audio_visualizer


# All 22 visualization functions in order, with display names
VISUALIZATIONS = [
    ("Waveform", audio_visualizer.plot_waveform),
    ("Volume Envelope", audio_visualizer.plot_waveform_envelope),
    ("Spectrogram", audio_visualizer.plot_spectrogram),
    ("Mel Spectrogram", audio_visualizer.plot_mel_spectrogram),
    ("Chromagram", audio_visualizer.plot_chromagram),
    ("Tonnetz", audio_visualizer.plot_tonnetz),
    ("Spectral Centroid", audio_visualizer.plot_spectral_centroid),
    ("Spectral Bandwidth", audio_visualizer.plot_spectral_bandwidth),
    ("Spectral Rolloff", audio_visualizer.plot_spectral_rolloff),
    ("RMS Energy", audio_visualizer.plot_rms_energy),
    ("Zero Crossing Rate", audio_visualizer.plot_zero_crossing_rate),
    ("Onset Strength", audio_visualizer.plot_onset_strength),
    ("Beat Tracking", audio_visualizer.plot_beat_track),
    ("Tempogram", audio_visualizer.plot_tempogram),
    ("MFCCs", audio_visualizer.plot_mfcc),
    ("Spectral Contrast", audio_visualizer.plot_spectral_contrast),
    ("Harmonic/Percussive", audio_visualizer.plot_harmonic_percussive),
    ("Frequency Bands", audio_visualizer.plot_frequency_bands),
    ("Dynamic Range", audio_visualizer.plot_dynamic_range),
    ("Spectral Flatness", audio_visualizer.plot_spectral_flatness),
    ("Combined Dashboard", audio_visualizer.plot_combined_dashboard),
    ("3D Spectrogram", audio_visualizer.plot_3d_spectrogram),
]

TOTAL_STEPS = len(VISUALIZATIONS)

DPI_OPTIONS = {
    "Normal (150 DPI)": 150,
    "High (200 DPI)": 200,
    "Ultra (300 DPI)": 300,
}

AUDIO_FILETYPES = [
    ("Audio files", "*.mp3 *.wav *.flac *.ogg *.aac *.m4a *.wma"),
    ("MP3", "*.mp3"),
    ("WAV", "*.wav"),
    ("FLAC", "*.flac"),
    ("All files", "*.*"),
]

ENCOURAGEMENT_MESSAGES = [
    "This one takes a bit longer...",
    "Still crunching the numbers...",
    "Good things take time...",
    "Almost there on this one...",
    "Processing away...",
    "Doing the math so Claude doesn't have to...",
    "Turning sound into sight...",
    "Claude's going to love this...",
    "Patience is a virtue, visualizations are a gift...",
    "The more complex the music, the richer the picture...",
]

ENCOURAGE_DELAY = 10   # seconds before first message appears
ENCOURAGE_INTERVAL = 5  # seconds between message rotation


def open_folder(path):
    """Open a folder in the system file manager."""
    system = platform.system()
    if system == "Darwin":
        subprocess.Popen(["open", str(path)])
    elif system == "Windows":
        subprocess.Popen(["explorer", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class AudioVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Visualizer")
        self.root.resizable(False, False)

        self.audio_path = None
        self.output_dir = None
        self.zip_path = None
        self.running = False
        self._step_start_time = 0
        self._encourage_after_id = None
        self._encourage_index = 0

        self._build_ui()
        self._center_window()

    def _center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"+{x}+{y}")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=20)
        main.grid(sticky="nsew")

        # Title
        title = ttk.Label(
            main,
            text="Audio Visualizer",
            font=("Helvetica", 18, "bold"),
        )
        title.grid(row=0, column=0, columnspan=3, pady=(0, 2))

        subtitle = ttk.Label(
            main,
            text="Let Claude Hear Your Music",
            font=("Helvetica", 11),
        )
        subtitle.grid(row=1, column=0, columnspan=3, pady=(0, 15))

        # Audio file picker
        ttk.Label(main, text="Audio File:").grid(
            row=2, column=0, sticky="w", pady=4
        )
        self.file_var = tk.StringVar(value="No file selected")
        file_entry = ttk.Entry(
            main, textvariable=self.file_var, state="readonly", width=45
        )
        file_entry.grid(row=2, column=1, padx=5, pady=4, sticky="ew")
        ttk.Button(main, text="Browse...", command=self._browse_file).grid(
            row=2, column=2, pady=4
        )

        # Output folder picker
        ttk.Label(main, text="Output Folder:").grid(
            row=3, column=0, sticky="w", pady=4
        )
        self.output_var = tk.StringVar(value="(auto)")
        output_entry = ttk.Entry(
            main, textvariable=self.output_var, state="readonly", width=45
        )
        output_entry.grid(row=3, column=1, padx=5, pady=4, sticky="ew")
        ttk.Button(main, text="Browse...", command=self._browse_output).grid(
            row=3, column=2, pady=4
        )

        # Quality dropdown
        ttk.Label(main, text="Quality:").grid(
            row=4, column=0, sticky="w", pady=4
        )
        self.quality_var = tk.StringVar(value="Normal (150 DPI)")
        quality_combo = ttk.Combobox(
            main,
            textvariable=self.quality_var,
            values=list(DPI_OPTIONS.keys()),
            state="readonly",
            width=20,
        )
        quality_combo.grid(row=4, column=1, padx=5, pady=4, sticky="w")

        # Generate button
        self.generate_btn = ttk.Button(
            main,
            text="Generate Visualizations",
            command=self._start_generate,
            state="disabled",
        )
        self.generate_btn.grid(
            row=5, column=0, columnspan=3, pady=(15, 8), sticky="ew"
        )

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            main,
            variable=self.progress_var,
            maximum=TOTAL_STEPS,
            mode="determinate",
        )
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)

        # Status label
        self.status_var = tk.StringVar(value="Select an audio file to begin.")
        status_label = ttk.Label(
            main, textvariable=self.status_var, font=("Helvetica", 10)
        )
        status_label.grid(row=7, column=0, columnspan=3, pady=(4, 0))

        # Secondary encouragement label
        self.encourage_var = tk.StringVar(value="")
        encourage_label = ttk.Label(
            main, textvariable=self.encourage_var, font=("Helvetica", 9, "italic"),
            foreground="gray",
        )
        encourage_label.grid(row=8, column=0, columnspan=3, pady=(0, 4))

        # Open folder button (hidden until complete)
        self.open_btn = ttk.Button(
            main,
            text="Open Output Folder",
            command=self._open_output,
            state="disabled",
        )
        self.open_btn.grid(row=9, column=0, columnspan=3, pady=(8, 0), sticky="ew")

        main.columnconfigure(1, weight=1)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File", filetypes=AUDIO_FILETYPES
        )
        if path:
            self.audio_path = Path(path)
            self.file_var.set(self.audio_path.name)
            self.generate_btn.configure(state="normal")
            self.status_var.set("Ready to generate.")

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_var.set(path)
        else:
            self.output_var.set("(auto)")

    def _start_generate(self):
        if self.running:
            return

        self.running = True
        self.generate_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Loading audio...")

        thread = threading.Thread(target=self._worker, daemon=True)
        thread.start()

    def _worker(self):
        try:
            import librosa
            import numpy as np

            # Set DPI
            dpi = DPI_OPTIONS[self.quality_var.get()]
            audio_visualizer.FIGURE_DPI = dpi

            # Load audio
            y, sr = audio_visualizer.load_audio(str(self.audio_path))

            # Determine output directory
            output_text = self.output_var.get()
            if output_text == "(auto)":
                output_dir = audio_visualizer.create_output_dir(self.audio_path)
            else:
                output_dir = Path(output_text)
                output_dir.mkdir(parents=True, exist_ok=True)

            self.output_dir = output_dir
            self.root.after(0, self._show_output_dir, str(output_dir))

            # Generate each visualization
            for i, (name, func) in enumerate(VISUALIZATIONS):
                step = i + 1
                self.root.after(
                    0,
                    self._update_progress,
                    step,
                    f"Generating: {name} ({step}/{TOTAL_STEPS})",
                )

                # Combined dashboard needs base_path as extra arg
                if func is audio_visualizer.plot_combined_dashboard:
                    func(y, sr, output_dir, self.audio_path)
                else:
                    func(y, sr, output_dir)

                # Free matplotlib/Core Graphics resources between figures
                plt.close('all')
                gc.collect()

            # Create guide
            self.root.after(0, self._update_progress, TOTAL_STEPS, "Creating guide...")
            duration = librosa.get_duration(y=y, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            audio_visualizer.create_visualization_guide(
                output_dir, duration, tempo, self.audio_path.stem
            )

            # Create zip archive, then move it into the output folder
            self.root.after(0, self._update_progress, TOTAL_STEPS, "Creating zip for Claude upload...")
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                tmp_stem = Path(tmp) / output_dir.name
                zip_path = shutil.make_archive(str(tmp_stem), 'zip', output_dir)
                final_zip = output_dir / Path(zip_path).name
                shutil.move(zip_path, final_zip)
            self.zip_path = final_zip

            self.root.after(0, self._on_complete)

        except Exception as e:
            self.root.after(0, self._on_error, str(e))

    def _show_output_dir(self, path):
        self.output_var.set(path)
        self.open_btn.configure(state="normal")

    def _update_progress(self, step, message):
        self.progress_var.set(step)
        self.status_var.set(message)
        self._reset_encouragement()

    def _reset_encouragement(self):
        """Reset the encouragement timer for the current step."""
        if self._encourage_after_id is not None:
            self.root.after_cancel(self._encourage_after_id)
            self._encourage_after_id = None
        self.encourage_var.set("")
        self._encourage_index = 0
        self._step_start_time = time.monotonic()
        if self.running:
            self._encourage_after_id = self.root.after(
                ENCOURAGE_DELAY * 1000, self._show_encouragement
            )

    def _show_encouragement(self):
        if not self.running:
            return
        msg = ENCOURAGEMENT_MESSAGES[self._encourage_index % len(ENCOURAGEMENT_MESSAGES)]
        self.encourage_var.set(msg)
        self._encourage_index += 1
        self._encourage_after_id = self.root.after(
            ENCOURAGE_INTERVAL * 1000, self._show_encouragement
        )

    def _stop_encouragement(self):
        if self._encourage_after_id is not None:
            self.root.after_cancel(self._encourage_after_id)
            self._encourage_after_id = None
        self.encourage_var.set("")

    def _on_complete(self):
        self.running = False
        self._stop_encouragement()
        self.progress_var.set(TOTAL_STEPS)
        zip_name = self.zip_path.name if self.zip_path else ""
        self.status_var.set(f"Done! {zip_name} is ready to upload to Claude.")
        self.generate_btn.configure(state="normal")
        self.open_btn.configure(state="normal")

    def _on_error(self, message):
        self.running = False
        self._stop_encouragement()
        self.status_var.set("Error occurred.")
        self.generate_btn.configure(state="normal")
        messagebox.showerror("Error", f"Visualization failed:\n\n{message}")

    def _open_output(self):
        if self.output_dir and self.output_dir.exists():
            open_folder(self.output_dir)


def main():
    root = tk.Tk()
    AudioVisualizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
