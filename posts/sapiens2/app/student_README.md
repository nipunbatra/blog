# Sapiens2 try-it API — student / collaborator client

You don't need any of the server code. Everything you need is one Python file.

## 1. Get the client

On Bhaskar (or any machine that can read `/DATA`):

```bash
cp /DATA/nipun.batra/sapiens2/client/student_client.py .
```

If you're on your own laptop, scp it across:

```bash
scp bhaskar.iitgn.ac.in:/DATA/nipun.batra/sapiens2/client/student_client.py .
```

## 2. Install deps

```bash
pip install requests tqdm
```

## 3. Reach the server

The API is at `http://10.0.62.159:8511` on the IITGN intranet. You're good if:

- you're on `IITGN-Wifi` / `eduroam` on campus, **or**
- you're connected to the IITGN VPN, **or**
- you SSH-tunnel from your laptop:

  ```bash
  ssh -L 18511:localhost:8511 bhaskar.iitgn.ac.in
  # then use http://localhost:18511 instead
  ```

Tell the script where the server is (default is the intranet URL):

```bash
export SAPIENS2_API=http://10.0.62.159:8511          # campus / VPN
export SAPIENS2_API=http://localhost:18511           # SSH tunnel
```

## 4. Run

One image, summary printed:

```bash
python student_client.py path/to/photo.jpg --task pose
```

A whole folder (recursive walk; resume-aware):

```bash
python student_client.py /path/to/images/ --task pose --workers 4 --out results.jsonl
```

Re-running the same command picks up where it stopped. The output is JSONL:
one JSON object per line, with the full server response plus a `src` field
pointing at the input. The big base64 image strings are stripped to keep the
file small.

## 5. Tasks available

- `pose`     — 308 keypoints (body + face + hands + feet)
- `seg`      — 29-class body-part segmentation (dome29 palette)
- `normal`   — disabled in this build
- `pointmap` — disabled in this build

## 6. Throughput

The server runs Sapiens2 1B on a single A5000, ~0.88 image/s. Practical
estimates:

|   N images | Time           |
|-----------:|----------------|
|        100 | ~ 2 minutes    |
|      1,000 | ~ 19 minutes   |
|      7,000 | ~ 2 h 13 min   |

`--workers` higher than 4 won't help — the GPU is the bottleneck. If you
need more throughput, ping Nipun about running a second API instance on
the second GPU, or switching to the smaller 0.4B model.

## 7. Reading the JSONL output

```python
import json
records = [json.loads(line) for line in open("results.jsonl")]

# Pose example
pose = next(r for r in records if r.get("task") == "pose")
for kp in pose["keypoints"]:
    if kp["score"] >= 0.5:
        print(f"{kp['name']:20s}  ({kp['x']:.0f}, {kp['y']:.0f})  score={kp['score']:.2f}")
```

## 8. Troubleshooting

- `can't reach <URL>`: not on the IITGN network / VPN / tunnel. See step 3.
- `503 disabled`: you asked for `normal` or `pointmap`; only `pose` and
  `seg` are enabled in this build.
- `500` with traceback: server-side bug; copy the JSON body and send it to
  Nipun.
