# GalaxEye — Backend Engineer, ML Systems
### Take-Home Assignment — Task 3: Problem-Solving

# Problem-Solving

## 1. The Classifier Is Wrong About 30% of the Time — What Do You Do, and How Do You Decide If It's "Good Enough"?

First, it matters *what kind* of wrong — a confusion matrix, not just an accuracy number. Confusing forest with agricultural land is a much smaller problem than confusing water with residential; the cost of an error is not uniform across class pairs, and "30% wrong" hides that entirely.

This is not hypothetical: the shipped classifier here is wrong 16% of the time overall on the real evaluation set, and the confusion matrix shows why — almost all of it is Highway–River confusion, since both are narrow, elongated, similarly toned features that plain colour and texture genuinely cannot always tell apart. That is a legible, explainable failure mode, not noise, and it points to exactly where effort should go if this were to be improved: better shape/linearity features, not "more data in general."

"Good enough" is then a question about the downstream use, not the model itself:

- **If an analyst reviews every flagged tile before acting**, an imperfect model with well-calibrated confidence — one that knows when it is guessing, even if the guess is often wrong — can already save real review time. It does not need to be right; it needs to correctly rank "check this one first." That is the case here: split by the model's own confidence flag, it is 95.7% accurate on the tiles it is confident about, and only 41.3% accurate on the roughly 22% it flags as low-confidence. If an analyst only had to hand-check that flagged 22%, the other 78% would already be reliable.
- **If a decision is made automatically off the label with no human in the loop**, that same 95.7%/41.3% split is the number that actually matters, not the blended 84% — and automation should be scoped to the high-confidence slice only.

Concretely, the steps are: check the confusion matrix for the worst class pairs (done — it is Highway/River here); check whether errors cluster at low confidence, which they do here, meaning the threshold is doing its job, or whether the model is confidently wrong, which it mostly is not; and only then decide between better features for the specific confused classes, more training data, or simply accepting the error rate and leaning on the human-review workflow for the flagged slice.

## 2. This Service Runs Offline, Unmonitored — A Month Later, How Would You Know It's Still Working?

Nothing calls home, so the only evidence is what is already in the SQLite store, plus whatever comes back when someone eventually looks. The `/stats` endpoint (already built) is designed to catch the two failure modes that matter most for a system like this:

- **It stopped running at all** — no new rows since some expected cadence. The cheapest possible check is the `ingested_at` timestamp on the most recent row.
- **It is running but drifting or broken** — the class distribution has shifted in a way that does not match what the terrain being surveyed should produce (for example, suddenly 90% of tiles labelled "cloud," which means either the sensor is malfunctioning or there has been a month of bad weather — either way, worth knowing), or average confidence has dropped, or the low-confidence rate has climbed. `/stats` already returns per-class count and average confidence for exactly this reason.

None of this requires live monitoring. Someone visiting the box once a month, or an offline script diffing this month's `/stats` output against last month's, catches both failure modes from data already being collected. The model version is also logged with every prediction, so if the model file ever silently changes or reverts, that is visible in the data too.

## 3. Tiles Are Ingesting Fine, but the Stored Results Look Wrong — Walk Through the Steps, in Order

1. **Reproduce on one tile.** Pick a specific row that looks wrong, find the source tile file, and re-run it through the classifier directly, bypassing the API, to see whether the same stored result comes back. This splits the problem in half immediately: a *different* result locally than what is stored points to a bug in ingestion or storage, not the model; the *same* wrong result points to a bug in features, the model, or training.

2. **If it is a storage-path bug:** check the `model_version` field on the bad rows — is this happening only for tiles ingested during or after a model swap, suggesting a stale singleton or the wrong file loaded? Check whether the stored `class_probabilities` JSON actually matches the stored `label` — a bug that picks the argmax from the wrong array would produce a mismatch that is easy to spot this way.

3. **If it is a features or model bug:** print the raw feature vector for the bad tile and sanity-check it by eye against expectations — is a water tile's feature vector actually blue-dominant? If not, the bug is likely in `features.py`; check the resize, normalize, and channel-order logic first, since a silently swapped RGB/BGR channel order is a classic version of exactly this symptom. If the feature vector looks right but the model still gets it wrong, that is a training-data or model-quality problem rather than a bug — back to Question 1.

4. **Check for a pattern across bad rows**, not just the one tile — same filename prefix, same ingestion time window, same predicted label. A pattern narrows the cause to a specific code path or a specific batch of input tiles, such as a different image format or colour mode from the rest, rather than a general model weakness.

5. **Check the obvious, boring culprits before anything fancier:** the wrong file being read (a path bug), an image loaded in the wrong colour mode (a CMYK tile decoded as if it were RGB), a stale cached model object, or a clock/timezone bug making `ingested_at` misleading rather than the label actually being wrong.

## 4. What's the Weakest Part of the Design, and What Would Break It First?

The classifier's feature set. Colour and texture statistics give real separation for most classes — Forest, Residential, Industrial, and SeaLake all land above 90% precision or recall on the real evaluation set — but they structurally cannot distinguish elongated linear features from each other by colour alone. Highway and River are both narrow, similarly toned shapes, and that is exactly where the model breaks: 8 of 30 Highway tiles are labelled River, and 10 of 30 River tiles are labelled Highway — by far the two worst cells in the entire confusion matrix.

It would break first, and worst, on any input dominated by that kind of thin linear structure — a canal, a road cutting through farmland, a runway — because nothing in the ten-dimensional feature vector captures shape or linearity, only colour and local pixel variance. The fix is not "more data" so much as "different features": something that captures the elongation or aspect ratio of the dominant structure in a tile, such as a cheap edge-orientation histogram, would target this specific weakness directly, rather than pursuing a generic accuracy push.
