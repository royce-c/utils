#!/bin/bash

ROOT="${1:-.}"

# for each .mp4.zip: unzip into a sibling subdir named after the zip stem,
# then transcribe the extracted MP4, writing all outputs into that same subdir
find "$ROOT" -type f -name "*.mp4.zip" | while IFS= read -r zipfile; do
    zip_dir=$(dirname "$zipfile")
    stem=$(basename "$zipfile" .mp4.zip)
    out_dir="$zip_dir/$stem"
    mkdir -p "$out_dir"

    # skip only if fully done (output MP4 with subtitles exists AND no progress marker)
    if [ ! -f "$out_dir/$stem.progress" ] && ffprobe -v error -select_streams s -show_entries stream=index -of csv=p=0 "$out_dir/$stem.mp4" 2>/dev/null | grep -q .; then
        echo "Skipping (already done): $out_dir/$stem.mp4"
        continue
    fi

    echo "Unzipping: $zipfile -> $out_dir"
    unzip -n "$zipfile" -d "$out_dir" || { echo "WARNING: failed to unzip $zipfile"; continue; }

    # find the extracted MP4 (there should be exactly one)
    mp4=$(find "$out_dir" -maxdepth 1 -type f -name "*.mp4" | head -n 1)
    if [ -z "$mp4" ]; then
        echo "WARNING: no .mp4 found in $out_dir after unzipping $zipfile"
        continue
    fi

    echo "Transcribing: $mp4 -> $out_dir"
    python3 transcribe.py "$mp4" "$out_dir" || echo "ERROR: failed to transcribe $mp4, continuing..."
done
