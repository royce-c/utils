#!/bin/bash

ROOT="${1:-.}"

# unzip all .mp4.zip files to their own directory
find "$ROOT" -type f -name "*.mp4.zip" | while read -r zipfile; do
    dir=$(dirname "$zipfile")
    echo "Unzipping: $zipfile -> $dir"
    unzip -n "$zipfile" -d "$dir"
done

# transcribe all .mp4 files, skip ones already done
find "$ROOT" -type f -name "*.mp4" | while read -r mp4; do
    base="${mp4%.mp4}"
    if [[ -f "${base}.srt" || -f "${base}.txt" ]]; then
        echo "Skipping (already done): $mp4"
        continue
    fi
    echo "Transcribing: $mp4"
    python transcribe.py "$mp4"
done
