import os, csv, subprocess, shlex
from pathlib import Path
from faster_whisper import WhisperModel

VODS=Path("vods"); OUT=Path("out"); OUT.mkdir(exist_ok=True, parents=True)
model = WhisperModel("small", device=os.getenv("ASR_DEVICE","cpu"), compute_type=os.getenv("ASR_COMPUTE","int8"))
rows=[]

def run(cmd): return subprocess.run(shlex.split(cmd), capture_output=True, text=True)

files = sorted([p for p in VODS.glob("*.mp4")
                if any(k in p.name.lower() for k in ("kcorp","karmine"))])
print("Fichiers sélectionnés:", len(files))
for v in files:

    tmp = OUT / (v.stem + "_sample.wav")
    # extrait 45s du début en mono 16k
    run(f'ffmpeg -y -i "{v}" -t 45 -vn -ac 1 -ar 16000 "{tmp}"')
    segments, info = model.transcribe(str(tmp))
    ctx = " ".join(s.text.strip() for s in segments)
    # titre via Ollama
    sys = "Generate one short French esports notification title. 70 chars max. No emojis."
    proc = subprocess.run(["ollama","run","llama3.1:8b"], input=f"{sys}\nTranscript:\n{ctx}\nTitle:", text=True, capture_output=True)
    title = (proc.stdout or "Titre").strip().splitlines()[0][:70]
    rows.append([v.name, 0, 45, title])
    try: tmp.unlink()
    except: pass

with open(OUT/"events.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["video","t_start","t_end","title"]); w.writerows(rows)
print("OK ->", OUT/"events.csv")
