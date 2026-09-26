# GalaxEye — Backend Engineer, ML Systems
### Take-Home Assignment — Task 1: Design Note

# Offline Satellite Tile Classification Service

## 1. What It Needs to Do

A tile arrives, gets classified by land-use type, and the result is stored so an analyst can later ask questions such as "show me everything flagged residential this week" or "what came in that the model wasn't sure about." All of this runs on hardware with no internet connection, and possibly with no one watching it in real time.

## 2. Components and Tile Flow

```
tile file → [Ingest API] → [Preprocess] → [Classifier] → [Result Store] → [Query API]
```

- **Ingest API (`POST /tiles`)** — accepts an image, does minimal validation (is it actually decodable as an image?), and hands it off.
- **Preprocess / feature extraction** — resizes, normalizes, and computes whatever the classifier needs. Kept as a separate module from the model itself so the two can be swapped independently (for example, a different classifier that expects a raw tensor instead of hand-engineered features).
- **Classifier** — loaded once at process startup and held in memory; scores one tile per request and returns not just a label but a full probability distribution over classes.
- **Result store** — a relational table with one row per tile: filename, timestamp, predicted label, confidence, full class-probability distribution (JSON), and model version. SQLite is used for this exercise; see Section 4 for why, and what would change at real scale.
- **Query API** — read endpoints over the store: filter by label, by confidence threshold, or by "low confidence only" (tiles needing human review), plus an aggregate statistics endpoint.

This is deliberately not built as a message queue / worker pool for this exercise — one process handling ingest synchronously is enough to prove the path end to end, and it is the honest complexity level for "offline box, modest tile volume." If tile volume or model latency grew, ingest and classification would split into a producer/consumer pair (for example, SQLite used as a durable queue, or a local task queue) so that a slow classification does not block accepting new tiles — noted here as a scaling seam rather than something built into this version.

## 3. The Model: What Was Chosen, and Why

The obvious default — a pretrained deep convolutional neural network such as ResNet or EfficientNet fine-tuned on EuroSAT — is a reasonable accuracy choice but a poor fit for the stated constraint that the system must run on isolated, offline hardware:

- Fetching pretrained weights the first time requires internet access, which contradicts the deployment target.
- A multi-hundred-megabyte checkpoint has to be vendored onto the box some other way (manual copy, container image), which is a real operational cost worth naming even if it is ultimately the right trade-off.
- It is a black box to whoever is on call when a result looks wrong.

A small, fully self-contained pipeline was used instead: hand-engineered spectral and texture features (channel means and standard deviations, a greenness index, a blueness index, brightness, and edge density — the classic, inexpensive proxies for vegetation, water, built-up areas, bare soil, and cloud cover) feeding a Random Forest classifier. It trains in under a second, serializes to a small file that ships with the repository, and every prediction is explainable in a single sentence ("this tile is dark and blue, therefore water, with high confidence"). The honest cost is a lower accuracy ceiling than a CNN would offer — spectral colour alone will not reliably distinguish, for example, two different crop types. If accuracy mattered more than offline-first simplicity, a small CNN (MobileNet-scale) would be fine-tuned ahead of time on a connected machine and the frozen weights shipped instead — the same deployment story, with a better ceiling, at the cost of being more expensive to build and harder to debug.

### Trained and Verified on Real Data

The shipped model is trained on the real labelled tiles provided by GalaxEye (a seven-class EuroSAT subset: Forest, River, Residential, Industrial, AnnualCrop, SeaLake, and Highway). Evaluated against the separate held-out evaluation set, it achieves:

- **83.8% overall accuracy (176 of 210 tiles).**
- **95.7% accuracy on the roughly 78% of tiles the model is confident about.**
- **41.3% accuracy on the roughly 22% of tiles it flags as low-confidence.**

This split is the confidence-flagging mechanism doing exactly what it is meant to do — the model is usefully self-aware about when it is guessing, even though the guesses themselves are often wrong. The confusion matrix shows that the errors are not random: Highway and River tiles are confused with each other most often, since both are narrow, elongated, and similarly toned features that plain colour and texture struggle to tell apart. This is a legible, explainable failure mode rather than noise, and is the concrete example underlying the answer to Part 3, Question 1. The training script also keeps a synthetic-data fallback, clearly marked in the code, so that the repository still runs standalone without the dataset attached; this fallback is not what the shipped model was actually trained on.

## 4. Storage: What and How

### What to Store per Tile

Filename, ingestion timestamp, predicted label, confidence, the full probability distribution (not just the top label — an analyst asking "was this actually a toss-up between forest and agricultural land?" needs that), and a model version string. Model version matters the moment the model is retrained, since old and new predictions then remain distinguishable rather than being silently mixed together.

### What Was Decided Not to Be Stored

The tile's raw pixels are not stored. Re-running classification is cheap and deterministic, whereas storing every image blob is not, especially at scale. In a real deployment, the source path to the tile would be kept instead of the bytes themselves, assuming tiles live somewhere addressable on disk, so that a human could inspect the actual image behind a surprising prediction without doubling storage requirements.

### Why SQLite Rather Than PostgreSQL

The offline, single-node, "no one is tending this" framing makes a zero-administration embedded file database the right default: no service to keep alive, no separate backup story, and a file that is trivially copyable for a snapshot. It comfortably handles the query patterns described above, indexed on label, confidence, and timestamp, up to a large number of tiles. If the system later needed concurrent writers, geospatial queries such as bounding-box search over tile location, or multi-node access, a move to PostGIS would be the natural upgrade path — noted here rather than built, since it is not implied by the stated scenario.

### What "Querying the Results" Is Taken to Mean

Two things an analyst is likely to actually want: filtered listing ("show me class X above confidence Y") and an aggregate view ("how many of each class, what is the average confidence, how many need review"). Free-text or geospatial query was not built, since the exercise does not establish that tiles carry location metadata; the schema does include unused latitude and longitude columns as a hook for that once such metadata is real.

## 5. Handling Low-Confidence Predictions

Three options were considered, roughly in order of increasing complexity:

- **Threshold and flag** (the option implemented) — below a confidence cutoff, the prediction is stored anyway but marked as low-confidence, so a human review queue is simply a query for that flag. This is inexpensive, and keeps every tile in the store without losing information.
- **Threshold and reject** — refuse to store or return a label at all, forcing every low-confidence tile into a separate manual pipeline. This gives a cleaner guarantee ("nothing in the main table is a guess"), but discards the model's second-best guess, which is often still useful signal.
- **Abstain with a class hierarchy** — fall back to a coarser label, such as "vegetation" rather than committing to forest versus agricultural, when the top two classes are close. This is more sophisticated and requires the class taxonomy to be designed for it up front — more than this exercise calls for.

The first option was chosen: flag rather than hide. An analyst can always filter a low-confidence tile out, and nothing about a legitimately uncertain tile is lost. The confidence threshold of 0.55 is a named constant in the classifier module rather than a magic number scattered through the code — deliberately a starting estimate, pending real calibration against a labelled validation set (see Part 3, Question 1, for how that would actually be set).

## 6. Assumptions

- Tiles are single images classified independently, with no assumption of temporal sequence or neighbouring-tile context.
- "Offline" is interpreted as "no calls out during ingest, classification, or query," which the implementation satisfies; this does not extend to build time, since installing Python packages uses public package registries in the same way that any offline system's build step would use a local mirror in production.
- A single land-use label is assigned per tile, rather than multi-label or segmentation output, matching the phrase "classifies each one by land-use type" in the assignment prompt.

## 7. Questions to Ask GalaxEye

- The provided classes (Forest, River, Residential, Industrial, AnnualCrop, SeaLake, and Highway) come from EuroSAT — is that the actual taxonomy GalaxEye cares about in production, or a stand-in for this exercise? Real deployments often need finer or coarser categories than a benchmark dataset happens to define.
- What is the actual tile volume and rate in production — one-off uploads, or a steady stream that would need a queue rather than synchronous handling?
- Do tiles already carry geolocation and capture-time metadata, or would that need to be derived or attached within this service?
- Is "isolated hardware" a single box, or a fleet that would eventually need results aggregated somewhere centrally?
- What does "an analyst querying results" actually look like day to day — an API consumed by another internal tool, a command-line interface, or a simple dashboard? The answer changes how much investment the query surface warrants, versus keeping it as minimal as it is in this slice.
