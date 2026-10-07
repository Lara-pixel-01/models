import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from transformers import pipeline, AutoTokenizer
from pathlib import Path
import time
import psutil

path = Path(__file__).parent / "article_nasa.txt"
article = path.read_text(encoding="utf-8").strip()
words = len(article.split())

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
tokenizer.model_max_length = 1024

summariser = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer = tokenizer)
summariser("warm up", max_length=5, min_length=1, do_sample=False)

start = time.perf_counter()
summarised_text = summariser(article, max_length = 150, min_length= 100, do_sample = False, truncation = True)
latency = time.perf_counter() - start


summary = summarised_text[0]["summary_text"]
summary_words = len(summary.split())
mem_mb = psutil.Process().memory_info().rss / 1_000_000



print(f"Original text length: {words}")
print(f"Summary text length: {summary_words}")
print(f"Latency: {latency:.2f}s")
print(f"Process RAM: {mem_mb:.1f} mb")
print(f"Summary: {summary}")