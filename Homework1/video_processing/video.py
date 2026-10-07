import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline
from pathlib import Path
import time
import psutil
import numpy as np

classifier = pipeline("video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")

VIDEOS_DIR = Path(__file__).parent / "videos"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


vids = [v for v in VIDEOS_DIR.iterdir() if v.suffix.lower() == ".mp4"]

lines = []
all_times = []
all_top1_conf = []

for video in sorted(vids):
    start = time.perf_counter()
    preds = classifier(str(video), top_k=5)
    inference_time = time.perf_counter() - start

    top1 = preds[0]
    all_times.append(inference_time)
    all_top1_conf.append(top1["score"])

    lines.append(f"=== {video.name} ===")
    print(f"=== {video.name} ===")
    for p in preds:
        line = f"  {p['label']:30s} {p['score']:.3f}"
        print(line)
        lines.append(line)
    lines.append("")

mem_mb = psutil.Process().memory_info().rss / 1_000_000

print(f"Videos processed: {len(vids)}")
print(f"Avg inference time: {np.mean(all_times):.2f}s")
print(f"Avg top-1 confidence: {np.mean(all_top1_conf):.3f}")
print(f"Total time: {sum(all_times):.2f}s")
print(f"Process RAM: {mem_mb:.1f} mb")

(RESULTS_DIR / "predictions.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"Saved to results/predictions.txt")