from faster_whisper import WhisperModel
import sys
import os
import subprocess
import tempfile

if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]
if not os.path.exists(audio_file):
    print(f"Error: file not found: {audio_file}")
    sys.exit(1)

# check if subtitles are already embedded in the container
def has_embedded_subtitles(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())

if has_embedded_subtitles(audio_file):
    print(f"Skipping (subtitles already embedded): {audio_file}")
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

base = os.path.splitext(os.path.abspath(audio_file))[0]

with open(base + ".txt", "w") as f:
    for s in all_segments:
        f.write(s.text.strip() + "\n")


def fmt_time(t):
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


# write a temporary SRT, mux it into the MP4, then clean up
with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as tmp:
    tmp_srt = tmp.name
    for i, s in enumerate(all_segments, 1):
        tmp.write(f"{i}\n{fmt_time(s.start)} --> {fmt_time(s.end)}\n{s.text.strip()}\n\n")

tmp_mp4 = base + ".subtitled.mp4"
try:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", audio_file,
            "-i", tmp_srt,
            "-c", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            tmp_mp4,
        ],
        check=True,
    )
    os.replace(tmp_mp4, audio_file)
finally:
    os.unlink(tmp_srt)
    if os.path.exists(tmp_mp4):
        os.unlink(tmp_mp4)

print(f"Saved: {base}.txt and subtitles embedded in {audio_file}")
