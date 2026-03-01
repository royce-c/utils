from faster_whisper import WhisperModel
import sys
import os
import subprocess

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <audio_file>")
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

# check if the output MP4 already exists with subtitles embedded
def has_embedded_subtitles(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())

if os.path.exists(base + ".mp4") and has_embedded_subtitles(base + ".mp4"):
    print(f"Skipping (output already exists with subtitles): {base}.mp4")
    sys.exit(0)

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16",
    num_workers=4,
    cpu_threads=8,
)

segments, info = model.transcribe(
    audio_file,
    language="en",
    beam_size=10, # higher = more accurate, slower
    best_of=5,
    patience=2, # allow beam search to explore longer
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], # fallback if confidence low
    condition_on_previous_text=True, # use prior context for coherence
    repetition_penalty=1.1, # suppress looping/repeating artifacts
    no_speech_threshold=0.6, # skip truly silent segments
    log_prob_threshold=-1.0, # retry low-confidence segments with higher temp
    compression_ratio_threshold=2.4,
    vad_filter=True, # strip silence before transcription
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400,
    ),
    word_timestamps=True, # finer segment alignment
)

print(f"Detected language: {info.language} ({info.language_probability:.2f})")

all_segments = []
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    all_segments.append(segment)

if not all_segments:
    print("WARNING: no speech detected, skipping output.")
    sys.exit(0)

if os.path.exists(base + ".txt"):
    print(f"Skipping (already exists): {base}.txt")
else:
    with open(base + ".txt", "w") as f:
        for s in all_segments:
            f.write(s.text.strip() + "\n")


def fmt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


# write SRT file
srt_file = base + ".srt"
if os.path.exists(srt_file):
    print(f"Skipping (already exists): {srt_file}")
else:
    with open(srt_file, "w") as f:
        for i, s in enumerate(all_segments, 1):
            f.write(f"{i}\n{fmt_time(s.start)} --> {fmt_time(s.end)}\n{s.text.strip()}\n\n")

# mux SRT into the MP4 container, writing directly to the output dir
out_mp4 = base + ".mp4"
tmp_mp4 = base + ".subtitled.mp4"
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

print(f"Saved: {base}.txt, {base}.srt, {out_mp4}")
