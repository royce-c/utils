"""Transcribe an audio/video file and produce .txt, .srt, and a muxed .mp4 with embedded subtitles.

Usage: python transcribe.py <audio_file> [output_dir]

Supports resuming interrupted runs via a .progress sentinel file.
If the output .mp4 already has embedded subtitles the file is skipped entirely.
If a .srt exists from a prior complete transcription but the mux never finished, only the mux step is re-run.
"""

from faster_whisper import WhisperModel
import re
import sys
import os
import subprocess

def fmt_time(t):
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

def has_embedded_subtitles(path):
    """Return True if the file at *path* contains a subtitle stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())

def mux_subtitles(video_path, srt_path, out_path, tmp_path):
    """Mux *srt_path* into *video_path*, writing the result to *out_path*.

    Uses *tmp_path* as a scratch file; atomically replaces *out_path* on
    success.  Cleans up *tmp_path* on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", srt_path,
                "-c", "copy",
                "-c:s", "mov_text",
                "-metadata:s:s:0", "language=eng",
                tmp_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: ffmpeg mux failed:\n{result.stderr}", file=sys.stderr)
            return False
        os.replace(tmp_path, out_path)
        return True
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def parse_srt_progress(srt_path):
    """Read an SRT file and return (last_end_seconds, last_block_index)."""
    ts_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    )
    last_end_str = None
    last_index = 0
    with open(srt_path) as f:
        for line in f:
            m = ts_pattern.search(line)
            if m:
                last_end_str = m.group(2)
            else:
                stripped = line.strip()
                if stripped.isdigit():
                    last_index = int(stripped)
    if not last_end_str:
        return 0.0, 0
    h, rest = last_end_str.split(":", 1)
    mins, rest = rest.split(":", 1)
    secs, ms = rest.split(",")
    seconds = int(h) * 3600 + int(mins) * 60 + int(secs) + int(ms) / 1000
    return seconds, last_index

# arg handling
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <audio_file> [output_dir]")
    sys.exit(1)

audio_file = os.path.abspath(sys.argv[1])
if not os.path.exists(audio_file):
    print(f"Error: file not found: {audio_file}")
    sys.exit(1)

if len(sys.argv) >= 3:
    out_dir = os.path.abspath(sys.argv[2])
    os.makedirs(out_dir, exist_ok=True)
else:
    out_dir = os.path.dirname(audio_file)

stem = os.path.splitext(os.path.basename(audio_file))[0]
base = os.path.join(out_dir, stem)

txt_file      = base + ".txt"
srt_file      = base + ".srt"
out_mp4       = base + ".mp4"
tmp_mp4       = base + ".subtitled.mp4"
tmp_audio     = base + ".resume_clip.mp4"
progress_file = base + ".progress"

# skip/resume logic, ordered from cheapest to most expensive check
resume_after = 0.0
resume_index = 0

if os.path.exists(progress_file):
    # interrupted run: figure out where we left off
    print(f"Resuming interrupted transcription of {audio_file}")
    if os.path.exists(srt_file):
        resume_after, resume_index = parse_srt_progress(srt_file)
        if resume_after > 0:
            print(f"  Last written: {fmt_time(resume_after)} "
                  f"(block {resume_index})")
else:
    # fresh run: quick checks before doing any real work

    # check already fully done
    if os.path.exists(out_mp4) and has_embedded_subtitles(out_mp4):
        print(f"Skipping (already done): {out_mp4}")
        sys.exit(0)

    # check transcription finished but mux didn't
    if os.path.exists(srt_file) and os.path.getsize(srt_file) > 0:
        print("Transcript found without muxed MP4 — re-muxing only...")
        if mux_subtitles(audio_file, srt_file, out_mp4, tmp_mp4):
            print(f"Saved: {out_mp4}")
            sys.exit(0)
        else:
            sys.exit(1)

# mark this run as in-progress
open(progress_file, "w").close()

# resume: trim the source audio so whisper only processes what remains
transcribe_input = audio_file
if resume_after > 0.0:
    print(f"  Trimming audio from {resume_after:.3f}s onward...")
    trim = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(resume_after),
         "-i", audio_file, "-c", "copy", tmp_audio],
        capture_output=True, text=True,
    )
    if trim.returncode != 0:
        print(f"ERROR: ffmpeg trim failed:\n{trim.stderr}", file=sys.stderr)
        os.unlink(progress_file)
        sys.exit(1)
    transcribe_input = tmp_audio

# transcription
model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    num_workers=4,
    cpu_threads=8,
)

segments, info = model.transcribe(
    transcribe_input,
    language="en",
    beam_size=10,
    best_of=5,
    patience=2,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    condition_on_previous_text=True,
    repetition_penalty=1.1,
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    compression_ratio_threshold=2.4,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
    word_timestamps=True,
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

# incremental write — append so resume extends existing files
txt_f = open(txt_file, "a")
srt_f = open(srt_file, "a")

total_written = resume_index
try:
    for segment in segments:
        abs_start = segment.start + resume_after
        abs_end = segment.end + resume_after

        print(f"[{abs_start:.2f}s -> {abs_end:.2f}s] {segment.text}")
        total_written += 1

        txt_f.write(segment.text.strip() + "\n")
        txt_f.flush()

        srt_f.write(
            f"{total_written}\n"
            f"{fmt_time(abs_start)} --> {fmt_time(abs_end)}\n"
            f"{segment.text.strip()}\n\n"
        )
        srt_f.flush()
finally:
    txt_f.close()
    srt_f.close()
    if os.path.exists(tmp_audio):
        os.unlink(tmp_audio)

if total_written == 0:
    print("WARNING: no speech detected, skipping output.")
    os.unlink(progress_file)
    sys.exit(0)

# mux subtitles into the mp4 container
if not mux_subtitles(audio_file, srt_file, out_mp4, tmp_mp4):
    sys.exit(1)

os.unlink(progress_file)
print(f"Saved: {txt_file}, {srt_file}, {out_mp4}")
