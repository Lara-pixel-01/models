from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from PIL import Image
import time
import psutil

BASE = Path(__file__).parent
PHOTOS_DIR = BASE / "photos"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4").signatures["serving_default"]

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]



photos = [p for p in PHOTOS_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

all_confidences = []
all_times = []

for photo in sorted(photos):
    img = Image.open(photo).convert("RGB")
    w, h = img.size

    inp = tf.cast(tf.expand_dims(np.array(img.resize((192, 192))), 0), tf.int32)

    start = time.perf_counter()
    kp = model(inp)["output_0"].numpy()[0, 0]
    inference_time = time.perf_counter() - start

    plt.imshow(np.array(img))
    for y, x, c in kp:
        if c > 0.3:
            plt.plot(x * w, y * h, "ro")
    for a, b in SKELETON:
        if kp[a][2] > 0.3 and kp[b][2] > 0.3:
            plt.plot([kp[a][1]*w, kp[b][1]*w], [kp[a][0]*h, kp[b][0]*h], "c-")
    plt.axis("off")

    out_name = f"result_{photos.index(photo)+1}.png"
    plt.savefig(RESULTS_DIR / out_name, bbox_inches="tight")
    plt.close()

    avg_conf = kp[:, 2].mean()
    visible_pct = (kp[:, 2] > 0.3).mean() * 100

    all_confidences.append(avg_conf)
    all_times.append(inference_time)

    print(f"  avg confidence: {avg_conf:.2f}")
    print(f"  visible keypoints: {visible_pct:.1f}%")
    print(f"  inference time: {inference_time*1000:.1f} ms -> {out_name}\n")

mem_mb = psutil.Process().memory_info().rss / 1_000_000
print(f"Images processed: {len(photos)}")
print(f"Avg confidence: {np.mean(all_confidences):.2f}")
print(f"Avg inference time: {np.mean(all_times)*1000:.1f} ms")
print(f"Total time: {sum(all_times):.2f}s")
print(f"Process RAM: {mem_mb:.1f} mb")