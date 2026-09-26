# GalaxEye Take-Home — Backend Engineer, ML Systems

This repo contains the design note (`Task1\_Design\_Note.md`), the working
slice (`app/`), and the Part 3 written answers (`Task3\_Problem\_Solving.md`).

## What's implemented (Part 2 scope)

The core path, end to end, fully offline:

* `POST /tiles` — upload one tile image → extract features → classify with
a locally-loaded model → store the result in SQLite → return it.
* `GET /tiles` — query stored results (filter by `label`, `min\_confidence`,
`low\_confidence\_only`).
* `GET /tiles/{id}` — fetch one stored result.
* `GET /stats` — per-class counts and average confidence.
* `GET /health` — liveness probe.

**Stubbed / explicitly out of scope** (see design note for how each would
actually be built): batch/directory ingestion, auth, tile geolocation
metadata (the DB column exists but is unused), a real trained model on a
real labeled dataset, retraining/drift-correction pipeline, a UI.

## Why the model looks the way it does

No internet access is used anywhere on the request path or at startup —
there's no download of pretrained deep-net weights. Instead, `app/features.py`
computes a small, explainable 10-dim feature vector (channel means/stds, a
greenness/blueness index, brightness, edge density) and
`app/train\_classifier.py` trains a `RandomForestClassifier` on those
features, serialized once to `data/model.pkl`. See `Task1\_Design\_Note.md`
for the full reasoning and trade-offs.

**The shipped `data/model.pkl` is trained on the real labeled tiles
GalaxEye provided** (`candidate\_tiles/`, 7 classes — Forest, River,
Residential, Industrial, AnnualCrop, SeaLake, Highway), not on synthetic
data. Evaluated against the separate `eval\_set`/`eval\_labels.csv`: **83.8%
overall accuracy**, and — more relevant to the design — **95.7% accuracy
on the tiles the model is confident about, vs. 41.3% on the \~22% it flags
as low-confidence**. See `Task3\_Problem\_Solving.md` Q1 for what that split
means in practice, and Q4 for the specific failure mode it can't fully
catch (Highway vs. River).

## Run it

```bash
pip install -r requirements.txt

# 1. (Already done — data/model.pkl is checked in, trained on the real
#    provided tiles.) To retrain from scratch:
python3 app/train\_classifier.py --data-dir /path/to/candidate\_tiles
#    (omit --data-dir to fall back to a small synthetic dataset instead --
#    see the warning it prints; that's not what data/model.pkl ships with)

# 2. Start the API
uvicorn app.main:app --reload --port 8008
```

## Evaluate it

```bash
python3 app/evaluate.py \\
  --eval-dir /path/to/eval\_set \\
  --labels-csv /path/to/eval\_labels.csv
```

Prints overall accuracy, the accuracy split by confidence flag, and a
confusion matrix.

## Try it

```bash
# ingest a tile
curl -X POST http://127.0.0.1:8008/tiles -F "file=@sample\_tiles/forest\_tile\_001.png"

# ingest everything in sample\_tiles/ (one real tile per class, from eval\_set)
for f in sample\_tiles/\*.png; do
  curl -s -X POST http://127.0.0.1:8008/tiles -F "file=@$f"; echo
done

# query
curl "http://127.0.0.1:8008/tiles?label=River"
curl "http://127.0.0.1:8008/tiles?low\_confidence\_only=true"
curl "http://127.0.0.1:8008/stats"
```

`sample\_tiles/` holds seven real tiles copied from the provided
`eval\_set/` (one per class, filenames say which). Point the same
`curl -F "file=@..."` commands at any other tile from the assignment zip
and it works unchanged — ingestion only assumes "readable image."

Interactive API docs: `http://127.0.0.1:8008/docs` (FastAPI's built-in
Swagger UI — works offline too, it's bundled, not CDN-loaded... actually
FastAPI's default docs page loads Swagger UI assets from a CDN. If the
target machine is truly air-gapped, use `/redoc` alternative or just the
curl examples above — flagged here as exactly the kind of "looks offline
but has one CDN dependency" bug described in Part 3, Q3.)

## Repo layout

```
Task1\_Design\_Note.md      Part 1
Task3\_Problem\_Solving.md  Part 3
README.md                 this file (Part 2 run instructions)
requirements.txt
app/
  features.py              feature extraction (shared by training + inference)
  train\_classifier.py       offline training script → data/model.pkl
  classifier.py             inference wrapper (load model, predict + confidence)
  db.py                     SQLite storage + query layer
  main.py                   FastAPI app / endpoints
  evaluate.py               scores the model against eval\_set/eval\_labels.csv
data/
  model.pkl                  trained classifier (checked in, ready to run)
sample\_tiles/                7 real tiles from eval\_set, one per class
```

