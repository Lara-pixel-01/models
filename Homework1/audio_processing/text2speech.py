from kokoro import KPipeline
import sounddevice as sd
from pathlib import Path
import time
import psutil

pipeline = KPipeline(lang_code="b")

path = Path(__file__).parent / "jekyll_hyde_book.txt"
text =  path.read_text(encoding="utf-8").strip()
words = len(text.split())

generator = pipeline(text, voice='bm_george')

start = time.perf_counter()
audio_seconds = 0
chunks = 0

for gs, ps, audio in generator:
    audio_seconds += len(audio) / 24000
    chunks += 1
    # sd.play(audio, samplerate=24000)
    # sd.wait() 


latency = time.perf_counter() - start
rtf = latency / audio_seconds
mem_mb = psutil.Process().memory_info().rss / 1_000_000


print(f"Original text length: {words}")
print(f"Chunks: {chunks}")
print(f"Audio duration: {audio_seconds:.2f}s")
print(f"Latency: {latency:.2f}s")
print(f"RTF: {rtf:.3f}")
print(f"RAM: {mem_mb:.1f} mb")