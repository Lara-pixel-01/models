import requests

while True:
    prompt = input(">>> ")
    if prompt.strip() == "\\bye":
        break
    if not prompt.strip():
        continue

    response = requests.post("http://localhost:11434/api/generate",json={"model": "qwen3:4b", "prompt": prompt, "stream": False, "think": True},)
    data = response.json()

    print(f"\nAnswer:\n{data['response'].strip()}\n")

requests.post("http://localhost:11434/api/generate", json={"model": "qwen3:4b", "prompt": "", "keep_alive": 0},)

print("Stopped Model")