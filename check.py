#!/usr/bin/env python3
"""
Movie Bot automatic launcher.

The current bot.py already checks the configured Google Drive input folder.
If there are no new videos, bot.py exits without doing anything.
If a video is present, bot.py processes it normally.
"""

import subprocess
import sys


def main() -> int:
    print("Checking Google Drive input folder...")
    print("Starting bot.py; it will process only new videos found in the input folder.")
    return subprocess.call([sys.executable, "bot.py"])


if __name__ == "__main__":
    raise SystemExit(main())
