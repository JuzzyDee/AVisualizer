#!/usr/bin/env python3
"""
Build helper for Audio Visualizer.
Runs PyInstaller to create a standalone application in dist/AudioVisualizer/.
"""

import subprocess
import sys


def main():
    print("Building Audio Visualizer...")
    print("This may take a few minutes.\n")

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "audio_visualizer_gui.spec", "--clean"],
        cwd=sys.path[0] or ".",
    )

    if result.returncode == 0:
        print("\nBuild complete! App is in dist/AudioVisualizer/")
    else:
        print("\nBuild failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
