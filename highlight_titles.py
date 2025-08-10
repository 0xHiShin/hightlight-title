import os, subprocess, shlex
from pathlib import Path
from datetime import timedelta
import numpy as np
import soundfile as sf
import pandas as pd
from faster_whisper import WhisperModel

# ==== Config ====
VODS_DIR = Path("vods")
OUT_DIR = Path("out"); (OUT_DIR/"clips").mkdir(parents=True, exist_ok=True)
AUDIO_DIR = OUT_DIR/"audios"; AUDIO_DIR.mkdir(parents=True, exist_ok=True)

ASR_SIZE = os.getenv("ASR_MODEL_SIZE","small")   # small|medium|large-v3
DEVICE   = os.getenv("ASR_DEVICE","cpu")         # cpu|cuda
CTYPE    = os.getenv("ASR_COMPUTE","int8")       # int8|int8_float16|float16
LLM_TAG  = os.getenv("LLM_MODEL","llama3.1:8b")
TITLE_LEN= int(os.getenv("TITLE_LEN","110"))
WIN_BEFORE = float(os.getenv("WIN_BEFORE","6"))
WIN_AFTER  = float(os.getenv("WIN_AFTER","6"))
PEAK_ZSCORE = float(os.getenv("PEAK_Z","1.3"))
MAX_EVENTS_PER_VOD = int(os.getenv("MAX_EVENTS","1"))  # 1 évènement => 10 titres par VOD

# Filtre fichiers
def list_vods():
    files = [p for p in VODS_DIR.iterdir() if p.suffix.lower() in {".mp4",".mkv",".mov",".m4v",".webm"}]
    return [p for p in files if any(k in p.name.lower() for k in ("karmine","kcorp"))]

KW = [k.lower() for k in """
pentakill quadra triple ace clutch vol de baron baron steal nashor drake dragon elder
inhibiteur nexus tour backdoor teamfight shutdown 1v3 1v4 1v5
""".split()]

def run(cmd, inp=None):
    return subprocess.run(shlex.split(cmd), input=inp, text=True, capture_output=True)

def ts(sec): return str(timedelta(seconds=round(sec)))

def ensure_wav(video: Path) -> Path:
    wav = AUDIO_DIR / f"{video.stem}.wav"
    if not wav.exists():
        r = run(f'ffmpeg -y -i "{video}" -vn -ac 1 -ar 16000 "{wav}"')
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg a échoué pour {video.name}: {r.stderr[:200]}")
    return wav

def rms_envelope(wav: Path, hop=0.20, win=0.40):
    y, sr = sf.read(str(wav), always_2d=False)
    if y.ndim > 1: y = y[:,0]
    hop_samp = max(1, int(sr*hop)); win_samp = max(1, int(sr*win))
    rms=[]
    for i in range(0, max(0,len(y)-win_samp), hop_samp):
        seg = y[i:i+win_samp]
        val = float(np.sqrt(np.mean(seg*seg)+1e-12)); t = i/sr
        rms.append((t,val))
    return rms

def detect_peaks(rms):
    if not rms: return []
    vals = np.array([v for _,v in rms]); mu=float(vals.mean()); sd=float(vals.std()+1e-12)
    cand=[(t,v) for t,v in rms if (v-mu)/sd >= PEAK_ZSCORE]
    cand.sort()
    merged=[]
    for t,v in cand:
        if not merged or t-merged[-1][0] > 8.0: merged.append([t,v])
        else:
            if v>merged[-1][1]: merged[-1]=[t,v]
    return [(t,v) for t,v in merged]

# Whisper instancié une seule fois
_ASR = WhisperModel(ASR_SIZE, device=DEVICE, compute_type=CTYPE)

import sys
def _audio_duration_sec(path: Path) -> float:
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=False)
    n = y.shape[0] if hasattr(y, "shape") else len(y)
    return float(n) / float(sr)

def transcribe(path: Path):
    dur = _audio_duration_sec(path)
    segments, info = _ASR.transcribe(
        str(path),
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    segs = []
    last_print = -1
    for s in segments:  # streaming
        segs.append(dict(start=float(s.start), end=float(s.end), text=s.text.strip()))
        pct = int(min(100, (segs[-1]["end"] / max(1e-6, dur)) * 100))
        # imprime seulement si le % a changé
        if pct != last_print:
            print(f"\r  Transcription: {pct:3d}% ({segs[-1]['end']:.1f}s / {dur:.1f}s)", end="", flush=True)
            last_print = pct
    print("")  # newline
    return segs, info.language


def contains_kw(txt):
    t = txt.lower()
    return any(k in t for k in KW)

def context_around(segs, t, pre=WIN_BEFORE, post=WIN_AFTER):
    s = max(0.0, t-pre); e = t+post
    # chevauchement, pas inclusion stricte
    ctx = " ".join(sg["text"] for sg in segs if sg["end"] > s and sg["start"] < e).strip()
    return s, e, ctx

def cut_clip(video: Path, s: float, e: float) -> Path:
    dur = max(1.0, e-s)
    out = OUT_DIR/"clips"/f"{video.stem}_{int(s)}_{int(e)}.mp4"
    run(f'ffmpeg -y -ss {s:.2f} -i "{video}" -t {dur:.2f} -c:v libx264 -preset veryfast -crf 23 -c:a aac -movflags +faststart "{out}"')
    return out

import subprocess, shlex, time

import json, subprocess, shlex, time

import ollama  # Assurez-vous d'ajouter cet import en haut de votre fichier
import json

def gen_titles(ctx: str):
    # On garde le même message système, c'est lui qui donne les instructions
    sys_msg = (
        "Tu es rédacteur de notifications d'esport League of Legends."
        f" Génère EXACTEMENT 10 titres courts en français, ≤ {TITLE_LEN} caractères."
        " Pas d'emojis, pas de hashtags, pas de guillemets."
        " Réponds UNIQUEMENT en JSON valide: {\"titles\":[\"t1\",\"t2\",...,\"t10\"]}"
        " Aucune autre phrase avant ou après."
    )
    
    # Et le même prompt utilisateur
    user_prompt = f'Transcript proche de l’action:\\n"""\\n{ctx}\\n"""\\nTitres:'

    titles = []
    try:
        # La nouvelle méthode, directe et propre !
        response = ollama.chat(
            model=LLM_TAG,
            messages=[
                {'role': 'system', 'content': sys_msg},
                {'role': 'user', 'content': user_prompt},
            ],
            format='json',  # On force Ollama à garantir une sortie JSON !
            options={'temperature': 0.7} # Un peu de créativité
        )
        
        # La librairie peut déjà retourner un objet JSON ou une chaîne JSON
        # On s'assure de bien avoir l'objet
        content_str = response['message']['content']
        json_data = json.loads(content_str)
        
        cand = json_data.get("titles", [])
        titles = [str(x)[:TITLE_LEN].strip() for x in cand if str(x).strip()]

    except Exception as e:
        # On ajoute un print pour voir l'erreur si elle se produit
        print(f"\n[Erreur de communication avec Ollama] : {e}")
        # La suite du code gèrera le cas où la liste est vide

    # Le reste de votre logique de fallback est parfait
    if not titles:
        titles = ["Action décisive"] * 10
    while len(titles) < 10:
        titles.append(titles[-1])
        
    return titles[:10]



def process_vod(v: Path):
    print(f"==> {v.name}")
    wav = ensure_wav(v)
    rms = rms_envelope(wav); print("  RMS frames:", len(rms))
    peaks = detect_peaks(rms); print("  Pics détectés:", len(peaks), f"(z≥{PEAK_ZSCORE})")
    segs, lang = transcribe(wav); print("  Segments ASR:", len(segs), "lang:", lang)

    events=[]
    for t,_ in peaks:
        s,e,ctx = context_around(segs,t)
        if contains_kw(ctx) or len(ctx.split())>=4:
            events.append((t,s,e,ctx))
    if not events and peaks:
        print("  Fallback: 2 plus gros pics")
        for t,_ in sorted(peaks,key=lambda x:x[1], reverse=True)[:2]:
            s,e,ctx = context_around(segs,t); events.append((t,s,e,ctx))
    if not events and segs:
        print("  Fallback: 1 évènement au milieu")
        mid=(segs[0]["start"]+segs[-1]["end"])/2
        s,e,ctx = context_around(segs,mid); events.append((mid,s,e,ctx))

    events = events[:MAX_EVENTS_PER_VOD]
    print("  Évènements retenus:", len(events))

    rows=[]
    for idx,(t,s,e,ctx) in enumerate(events,1):
        print(f"    -> evt #{idx} {ts(s)}–{ts(e)} | génération titres…")
        titles = gen_titles(ctx)
        clip = cut_clip(v,s,e)
        row={"video":v.name,"event_id":idx,"t_peak":round(t,2),
             "t_start":round(s,2),"t_end":round(e,2),"ts_window":f"{ts(s)} - {ts(e)}",
             "lang":lang,"clip_file":clip.name,"context":ctx}
        for i,tit in enumerate(titles,1): row[f"title_{i}"]=tit
        rows.append(row)
    return rows

def save_events(rows):
    df = pd.DataFrame(rows); csv_path = OUT_DIR/"events.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nOK -> {csv_path} | lignes: {len(df)}\nClips -> {OUT_DIR/'clips'}")

def main():
    videos = list_vods()
    print(f"{len(videos)} VOD: {[v.name for v in videos]}")
    if not videos:
        print("Aucune VOD correspondante (karmine/kcorp)")
        return
    all_rows=[]
    for v in videos:
        all_rows.extend(process_vod(v))
    save_events(all_rows)

if __name__ == "__main__":
    main()
