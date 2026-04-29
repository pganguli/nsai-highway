#!/usr/bin/env bash
# Usage: mp4_to_gif.sh <input.mp4> <output.gif>
# Converts an MP4 to a 2x-speed GIF at 480px width using a two-pass palette.
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input.mp4> <output.gif>" >&2
    exit 1
fi

INPUT="$1"
OUTPUT="$2"

ffmpeg -y -i "$INPUT" \
    -vf "setpts=0.5*PTS,fps=15,scale=480:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
    "$OUTPUT"
