#!/usr/bin/env python3
"""
Media File Picker (macOS + Windows)

- Detects OS (macOS / Windows)
- Opens native file selection dialog (Finder / File Explorer)
- Filters for common audio/video extensions
- Prints the selected file path to stdout

Python: 3.14+
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as e:
    raise RuntimeError(
        "tkinter is required but not available in this Python installation.\n"
        "On Windows: reinstall Python with 'tcl/tk' enabled.\n"
        "On macOS: use the official python.org installer (usually includes Tk)."
    ) from e


AUDIO_EXTS = [
    ".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg", ".opus", ".wma", ".aiff", ".alac",
]
VIDEO_EXTS = [
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg",
]

# Combined list for convenience
MEDIA_EXTS = sorted(set(AUDIO_EXTS + VIDEO_EXTS))


def detect_os() -> str:
    """
    Returns a friendly OS label: 'Windows', 'macOS', 'Linux/Other'
    """
    system = platform.system()
    if system == "Windows":
        return "Windows"
    if system == "Darwin":
        return "macOS"
    return "Linux/Other"


def build_filetypes():
    """
    tkinter 'filetypes' filter:
    - A combined filter for audio/video
    - Separate audio-only and video-only
    - All files fallback
    """
    media_patterns = [f"*{ext}" for ext in MEDIA_EXTS]
    audio_patterns = [f"*{ext}" for ext in AUDIO_EXTS]
    video_patterns = [f"*{ext}" for ext in VIDEO_EXTS]

    return [
        ("Audio & Video", " ".join(media_patterns)),
        ("Audio", " ".join(audio_patterns)),
        ("Video", " ".join(video_patterns)),
        ("All files", "*.*"),
    ]


def choose_media_file(initial_dir: Path | None = None) -> Path | None:
    """
    Opens a native file selection dialog and returns the chosen Path, or None if cancelled.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes("-topmost", True)  # Bring dialog to front (best-effort)

    if initial_dir is None:
        initial_dir = Path.home()

    file_path = filedialog.askopenfilename(
        title="Select an audio or video file",
        initialdir=str(initial_dir),
        filetypes=build_filetypes(),
    )

    root.destroy()

    if not file_path:
        return None
    return Path(file_path).expanduser().resolve()


def main() -> int:
    os_label = detect_os()
    print(f"Detected OS: {os_label}")

    selected = choose_media_file()

    if selected is None:
        print("No file selected.")
        return 1

    # Extra safety: ensure it matches our extension list (user can still pick All files)
    if selected.suffix.lower() not in MEDIA_EXTS:
        # Not fatal—just warn
        print(
            f"Warning: Selected file extension '{selected.suffix}' is not in the media filter list.",
            file=sys.stderr,
        )

    print(str(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
