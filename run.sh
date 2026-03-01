#!/bin/bash

# Batch transcribe .mp4 files (from .mp4.zip archives or bare .mp4s).

# Usage: bash run.sh [--destructive] <root_dir>

# default (non-destructive): Creates a subdir per video, writes mp4/txt/srt there. Originals (.mp4.zip and .mp4) are left untouched.

# --destructive: removes original source mp4 and mp4.zip files after successful transcription.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DESTRUCTIVE=false

# parse arguments
usage() {
    echo "Usage: $0 [--destructive] <root_dir>"
    exit 1
}

while [[ "$1" == --* ]]; do
    case "$1" in
        --destructive) DESTRUCTIVE=true; shift ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

ROOT="${1:-.}"
if [ ! -d "$ROOT" ]; then
    echo "Error: directory not found: $ROOT"
    exit 1
fi
ROOT="$(realpath "$ROOT")"

echo "Mode: $([ "$DESTRUCTIVE" = true ] && echo "destructive" || echo "non-destructive")"
echo "Root: $ROOT"
echo "────────────────────────────────────────"

# check if an mp4 has embedded subtitles
has_subs() {
    ffprobe -v error -select_streams s \
        -show_entries stream=index -of csv=p=0 "$1" 2>/dev/null | grep -q .
}

# verify outputs and optionally delete originals
verify_and_cleanup() {
    local out_mp4="$1" # path to the output mp4 in the subdir
    local source="$2" # path to the original file to delete (zip or mp4)

    if [ ! -f "$out_mp4" ]; then
        echo "  ✗ Verification failed: output not found: $out_mp4"
        return 1
    fi

    if ! has_subs "$out_mp4"; then
        echo "  ✗ Verification failed: no subtitles in $out_mp4"
        return 1
    fi

    echo "  ✓ Verified: $out_mp4"

    if [ "$DESTRUCTIVE" = true ] && [ -n "$source" ] && [ -f "$source" ]; then
        # safety: never delete the output itself
        local abs_out abs_src
        abs_out="$(realpath "$out_mp4")"
        abs_src="$(realpath "$source")"
        if [ "$abs_out" != "$abs_src" ]; then
            rm -f "$source"
            echo "  ✓ Deleted original: $source"
        fi
    fi
}

# process .mp4.zip files
find "$ROOT" -type f -name "*.mp4.zip" | sort | while IFS= read -r zipfile; do
    zip_dir="$(dirname "$zipfile")"
    stem="$(basename "$zipfile" .mp4.zip)"
    out_dir="$zip_dir/$stem"
    out_mp4="$out_dir/$stem.mp4"

    # skip if already done
    if [ ! -f "$out_dir/$stem.progress" ] && [ -f "$out_mp4" ] && has_subs "$out_mp4"; then
        echo "Skipping (done): $out_mp4"
        continue
    fi

    echo "── $zipfile ──"
    mkdir -p "$out_dir"

    # unzip (skips files that already exist)
    echo "  Unzipping..."
    if ! unzip -n "$zipfile" -d "$out_dir" > /dev/null; then
        echo "  WARNING: failed to unzip $zipfile"
        continue
    fi

    # find the extracted mp4 (the one without embedded subtitles)
    mp4=""
    while IFS= read -r candidate; do
        if ! has_subs "$candidate"; then
            mp4="$candidate"
            break
        fi
    done < <(find "$out_dir" -maxdepth 1 -type f -name "*.mp4")

    if [ -z "$mp4" ]; then
        echo "  WARNING: no unprocessed .mp4 found in $out_dir"
        continue
    fi

    # transcribe
    echo "  Transcribing: $mp4"
    if ! python3 "$SCRIPT_DIR/transcribe.py" "$mp4" "$out_dir"; then
        echo "  ERROR: transcription failed for $mp4, continuing..."
        continue
    fi

    # verify and optionally delete the zip
    verify_and_cleanup "$out_mp4" "$zipfile"
done

# process bare .mp4 files (not inside an output subdir)
# Only pick up .mp4 files that sit next to a .mp4.zip OR have no zip at all
# skip any .mp4 that is already inside an output subdir (i.e. already the result of a previous run).
find "$ROOT" -type f -name "*.mp4" | sort | while IFS= read -r mp4; do
    mp4_dir="$(dirname "$mp4")"
    stem="$(basename "$mp4" .mp4)"
    out_dir="$mp4_dir/$stem"
    out_mp4="$out_dir/$stem.mp4"

    # skip if this mp4 is inside an output subdir (parent dir name == stem)
    if [ "$(basename "$mp4_dir")" = "$stem" ]; then
        continue
    fi

    # skip if a corresponding .zip exists — the zip handler above owns it
    if [ -f "$mp4.zip" ]; then
        continue
    fi

    # skip if already done
    if [ ! -f "$out_dir/$stem.progress" ] && [ -f "$out_mp4" ] && has_subs "$out_mp4"; then
        echo "Skipping (done): $out_mp4"
        continue
    fi

    echo "── $mp4 ──"
    mkdir -p "$out_dir"

    # for bare mp4: move the original into the subdir as the source
    # (transcribe.py will mux subtitles and write out_mp4 over it)
    src_mp4="$out_dir/$stem.mp4"
    if [ ! -f "$src_mp4" ]; then
        cp "$mp4" "$src_mp4"
    fi

    # transcribe (source = the copy in the subdir)
    echo "  Transcribing: $src_mp4"
    if ! python3 "$SCRIPT_DIR/transcribe.py" "$src_mp4" "$out_dir"; then
        echo "  ERROR: transcription failed for $mp4, continuing..."
        continue
    fi

    # verify and optionally delete the original bare mp4
    verify_and_cleanup "$out_mp4" "$mp4"
done

echo "────────────────────────────────────────"
echo "Done."
