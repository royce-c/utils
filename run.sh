#!/bin/bash
set -e

ROOT="${1:-.}"

# unzip all .mp4.zip files to their own directory
find "$ROOT" -type f -name "*.mp4.zip" | while IFS= read -r zipfile; do
    dir=$(dirname "$zipfile")
    echo "Unzipping: $zipfile -> $dir"
    unzip -n "$zipfile" -d "$dir"
done

# transcribe all .mp4 files, skip ones that already have embedded subtitles
find "$ROOT" -type f -name "*.mp4" | while IFS= read -r mp4; do
    if ffprobe -v error -select_streams s -show_entries stream=index -of csv=p=0 "$mp4" 2>/dev/null | grep -q .; then
        echo "Skipping (subtitles already embedded): $mp4"
        continue
    fi
    echo "Transcribing: $mp4"
    python3 transcribe.py "$mp4"
done
