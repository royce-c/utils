from faster_whisper import WhisperModel
import re
import sys
import os
import subprocess


if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <audio_file> [output_dir]")
    sys.exit(1)

audio_file = sys.argv[1]
if not os.path.exists(audio_file):
    print(f"Error: file not found: {audio_file}")
    sys.exit(1)

# optional second argument: directory to write outputs into
if len(sys.argv) >= 3:
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
else:
    out_dir = os.path.dirname(os.path.abspath(audio_file))

stem = os.path.splitext(os.path.basename(audio_file))[0]
base = os.path.join(out_dir, stem)
txt_file = base + ".txt"
srt_file = base + ".srt"
out_mp4  = base + ".mp4"
tmp_mp4  = base + ".subtitled.mp4"
tmp_audio = base + ".resume_clip.mp4"  # trimmed clip used when resuming
progress_file = base + ".progress"  # exists if a previous run was interrupted


def fmt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


def has_embedded_subtitles(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


# --- resume state -----------------------------------------------------------
# A .progress file means the previous run was interrupted.
# Read the last end-timestamp written to the .srt so we can skip past it.

resume_after = 0.0 # skip segments whose end time is <= this
resume_index = 0 # next SRT block index to write (1-based)

if os.path.exists(progress_file):
    print(f"Resuming interrupted transcription of {audio_file}")
    # parse the last timestamp from the existing SRT
    if os.path.exists(srt_file):
        ts_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
        )
        last_end_str = None
        last_index = 0
        with open(srt_file) as f:
            for line in f:
                m = ts_pattern.search(line)
                if m:
                    last_end_str = m.group(2)
                else:
                    # SRT index lines are bare integers
                    stripped = line.strip()
                    if stripped.isdigit():
                        last_index = int(stripped)
        if last_end_str:
            h, rest = last_end_str.split(":", 1)
            mins, rest = rest.split(":", 1)
            secs, ms = rest.split(",")
            resume_after = int(h) * 3600 + int(mins) * 60 + int(secs) + int(ms) / 1000
            resume_index = last_index  # next block continues from here
            print(f"  Resuming after {last_end_str} (SRT block {last_index})")
else:
    # fresh run — check if already fully done
    if os.path.exists(out_mp4) and has_embedded_subtitles(out_mp4):
        print(f"Skipping (already done): {out_mp4}")
        sys.exit(0)
    # if .srt already exists and has content, the transcription is complete but
    # the mux step never finished — skip straight to muxing without re-transcribing
    if os.path.exists(srt_file) and os.path.getsize(srt_file) > 0:
        print(f"Transcript already complete, re-muxing subtitles into MP4...")
        open(progress_file, "w").close()
        try:
            result = subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", audio_file,
                    "-i", srt_file,
                    "-c", "copy",
                    "-c:s", "mov_text",
                    "-metadata:s:s:0", "language=eng",
                    tmp_mp4,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"ERROR: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
                sys.exit(1)
            os.replace(tmp_mp4, out_mp4)
        finally:
            if os.path.exists(tmp_mp4):
                os.unlink(tmp_mp4)
        os.unlink(progress_file)
        print(f"Saved: {out_mp4}")
        sys.exit(0)

# mark this run as in-progress
open(progress_file, "w").close()

# ---------------------------------------------------------------------------
# If resuming, extract a trimmed clip from resume_after to end so whisper
# only processes the remaining audio. Timestamps are relative to the clip
# start, so we add resume_after back when writing.

transcribe_input = audio_file
if resume_after > 0.0:
    print(f"  Extracting audio from {resume_after:.3f}s for transcription...")
    trim_result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(resume_after),
            "-i", audio_file,
            "-c", "copy",
            tmp_audio,
        ],
        capture_output=True, text=True,
    )
    if trim_result.returncode != 0:
        print(f"ERROR: ffmpeg trim failed:\n{trim_result.stderr}", file=sys.stderr)
        os.unlink(progress_file)
        sys.exit(1)
    transcribe_input = tmp_audio

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
    beam_size=10, # higher = more accurate, slower
    best_of=5,
    patience=2, # allow beam search to explore longer
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], # fallback if confidence low
    condition_on_previous_text=True, # use prior context for coherence
    repetition_penalty=1.1, # suppress looping/repeating artifacts
    no_speech_threshold=0.6, # skip truly silent segments
    log_prob_threshold=-1.0, # retry low-confidence segments
    compression_ratio_threshold=2.4,
    vad_filter=True, # strip silence before transcription
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400,
    ),
    word_timestamps=True, # finer segment alignment
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

# open output files in append mode so resuming extends them
txt_f = open(txt_file, "a")
srt_f = open(srt_file, "a")

total_written = resume_index
try:
    for segment in segments:
        # timestamps from whisper are relative to transcribe_input;
        # add resume_after to convert back to absolute time in the original file
        abs_start = segment.start + resume_after
        abs_end   = segment.end   + resume_after

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
    # clean up the trimmed clip if one was created
    if os.path.exists(tmp_audio):
        os.unlink(tmp_audio)

if total_written == 0:
    print("WARNING: no speech detected, skipping output.")
    os.unlink(progress_file)
    sys.exit(0)

# mux completed SRT into the MP4 container
# If the source and output are the same path (e.g. re-processing an already-extracted MP4),
# write to tmp_mp4 first then atomically replace — os.replace handles this correctly
# since it's on the same filesystem, but we must not unlink tmp_mp4 if it IS out_mp4.
try:
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", audio_file,
            "-i", srt_file,
            "-c", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            tmp_mp4,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: ffmpeg failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    os.replace(tmp_mp4, out_mp4)
finally:
    if os.path.exists(tmp_mp4):
        os.unlink(tmp_mp4)

# clean run complete — remove the progress marker
os.unlink(progress_file)

print(f"Saved: {txt_file}, {srt_file}, {out_mp4}")
