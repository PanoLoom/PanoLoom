# Synthetic ground-truth datasets

`generate.py` renders virtual camera shots from a CC0 equirectangular
panorama (Poly Haven) at exactly known yaw/pitch/roll/HFOV, so alignment
accuracy is measurable in degrees instead of eyeballed.

```sh
# one-time: fetch a source pano (CC0, tonemapped 8k JPG)
../.venv/bin/python generate.py --fetch kloppenheim_06

# presets: pair (2 shots), ring (8 shots, 1 row), sphere (26 shots, 3 rows
# + zenith + nadir, portrait orientation)
../.venv/bin/python generate.py --set ring   --equirect assets/kloppenheim_06.jpg
../.venv/bin/python generate.py --set sphere --equirect assets/kloppenheim_06.jpg --ev-jitter 0.4
```

Each set is written to `generated/<name>/` with a `ground_truth.json`
holding per-shot poses and the exact camera/rotation conventions (which
MUST match panoloom-core — they are documented in the file itself).

Everything here is reproducible (fixed seed) and gitignored.
