#!/usr/bin/env python3
"""Synthetic ground-truth dataset generator.

Renders virtual camera shots from a CC0 equirectangular panorama at known
yaw/pitch/roll/HFOV, so the engine's recovered poses can be scored in degrees
against ground_truth.json.

Conventions (MUST match panoloom-core; also recorded in ground_truth.json):
  - Camera frame: x right, y down, z forward (OpenCV convention).
  - Rotation world<-camera: R = Ry(yaw) @ Rx(pitch) @ Rz(roll), angles in
    degrees. Positive yaw looks right (toward +x), positive pitch looks UP,
    positive roll tilts clockwise in the image.
  - World direction -> spherical: lon = atan2(x, z), lat = asin(-y)
    (lat positive above the horizon).
  - Equirect pixel: u = (lon / 2pi + 0.5) * W, v = (0.5 - lat / pi) * H.

Usage:
  python generate.py --fetch kloppenheim_06                 # download source pano
  python generate.py --set ring --equirect assets/klop.jpg  # 8-shot single row
  python generate.py --set sphere --equirect assets/klop.jpg --ev-jitter 0.4
"""

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
GENERATED = HERE / "generated"

SEED = 20260810  # fixed: datasets must be bit-reproducible


def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def camera_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """world<-camera rotation. Positive pitch looks up (y is down)."""
    return rot_y(yaw) @ rot_x(-pitch) @ rot_z(roll)


def render_view(
    equirect: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    hfov_deg: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    src_h, src_w = equirect.shape[:2]
    # Pad one column so bilinear sampling across the lon=+/-pi seam works.
    padded = np.hstack([equirect, equirect[:, :1]])

    tan_half_h = math.tan(math.radians(hfov_deg) / 2)
    tan_half_v = tan_half_h * out_h / out_w

    # Pixel-center rays in the camera frame.
    xs = (2 * (np.arange(out_w) + 0.5) / out_w - 1) * tan_half_h
    ys = (2 * (np.arange(out_h) + 0.5) / out_h - 1) * tan_half_v
    xv, yv = np.meshgrid(xs, ys)
    dirs = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    world = dirs @ camera_rotation(yaw, pitch, roll).T
    lon = np.arctan2(world[..., 0], world[..., 2])
    lat = np.arcsin(np.clip(-world[..., 1], -1.0, 1.0))

    map_x = ((lon / (2 * math.pi) + 0.5) * src_w) % src_w
    map_y = np.clip((0.5 - lat / math.pi) * src_h, 0, src_h - 1)

    return cv2.remap(
        padded,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def apply_ev(img: np.ndarray, ev: float) -> np.ndarray:
    # Linear-light exposure shift through an approximate sRGB gamma of 2.2.
    linear = (img.astype(np.float32) / 255.0) ** 2.2
    shifted = np.clip(linear * (2.0**ev), 0.0, 1.0)
    return (shifted ** (1 / 2.2) * 255.0 + 0.5).astype(np.uint8)


def shot_list(preset: str, hfov: float) -> list[dict]:
    shots = []
    if preset == "ring":
        for i in range(8):
            shots.append({"yaw": i * 45.0, "pitch": 0.0, "roll": 0.0})
    elif preset == "sphere":
        for i in range(8):
            shots.append({"yaw": i * 45.0, "pitch": 0.0, "roll": 0.0})
        for row_pitch in (45.0, -45.0):
            for i in range(8):
                shots.append({"yaw": i * 45.0 + 22.5, "pitch": row_pitch, "roll": 0.0})
        shots.append({"yaw": 0.0, "pitch": 90.0, "roll": 0.0})   # zenith
        shots.append({"yaw": 0.0, "pitch": -90.0, "roll": 0.0})  # nadir
    elif preset == "pair":
        shots = [
            {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            {"yaw": 30.0, "pitch": 0.0, "roll": 0.0},
        ]
    else:
        raise SystemExit(f"unknown preset: {preset}")
    for s in shots:
        s["hfovDeg"] = hfov
    return shots


def fetch_polyhaven(slug: str) -> Path:
    """Download the tonemapped JPG of a Poly Haven HDRI (CC0)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    dest = ASSETS / f"{slug}.jpg"
    if dest.exists():
        print(f"already have {dest}")
        return dest
    # Poly Haven blocks the default urllib user agent.
    def fetch(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "PanoLoom-testdata/0.1"})
        with urllib.request.urlopen(req) as r:
            return r.read()

    files = json.loads(fetch(f"https://api.polyhaven.com/files/{slug}"))
    url = files.get("tonemapped", {}).get("url")
    if not url:
        raise SystemExit(
            f"no tonemapped JPG listed for '{slug}' — pick another asset "
            f"or download an equirect JPG manually into {ASSETS}/"
        )
    print(f"downloading {url}")
    dest.write_bytes(fetch(url))
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", metavar="SLUG", help="download a Poly Haven HDRI's tonemapped JPG and exit")
    ap.add_argument("--equirect", type=Path, help="source equirectangular image")
    ap.add_argument("--set", dest="preset", default="ring", choices=["pair", "ring", "sphere"])
    ap.add_argument("--name", help="output set name (default: <preset>_<source stem>)")
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--width", type=int, default=1600, help="shot width in px")
    ap.add_argument("--height", type=int, default=1067)
    ap.add_argument("--ev-jitter", type=float, default=0.0, help="uniform random EV offset per shot, +/- this value")
    args = ap.parse_args()

    if args.fetch:
        fetch_polyhaven(args.fetch)
        return
    if not args.equirect:
        ap.error("--equirect is required (or use --fetch first)")

    equirect = cv2.imread(str(args.equirect), cv2.IMREAD_COLOR)
    if equirect is None:
        raise SystemExit(f"cannot read {args.equirect}")

    # Full-sphere coverage needs vertical overlap between rows 45° apart:
    # landscape at hfov 65 has only ~46° vfov (sub-1° overlap!), so the sphere
    # preset shoots portrait — exactly like real pano-head workflows.
    if args.preset == "sphere" and args.width > args.height:
        args.width, args.height = args.height, args.width

    name = args.name or f"{args.preset}_{args.equirect.stem}"
    out_dir = GENERATED / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    shots = shot_list(args.preset, args.hfov)
    truth: dict = {
        "conventions": {
            "cameraFrame": "x right, y down, z forward",
            "rotation": "R_world_from_cam = Ry(yaw) @ Rx(-pitch) @ Rz(roll), degrees; +pitch looks up",
            "spherical": "lon = atan2(x, z); lat = asin(-y)",
            "equirectPixel": "u = (lon/2pi + 0.5)*W; v = (0.5 - lat/pi)*H",
        },
        "source": args.equirect.name,
        "seed": SEED,
        "images": [],
    }

    for i, shot in enumerate(shots):
        view = render_view(
            equirect, shot["yaw"], shot["pitch"], shot["roll"],
            shot["hfovDeg"], args.width, args.height,
        )
        ev = float(rng.uniform(-args.ev_jitter, args.ev_jitter)) if args.ev_jitter else 0.0
        if ev:
            view = apply_ev(view, ev)
        file_name = f"img_{i:03d}.jpg"
        cv2.imwrite(str(out_dir / file_name), view, [cv2.IMWRITE_JPEG_QUALITY, 95])
        truth["images"].append({
            "fileName": file_name,
            "width": args.width,
            "height": args.height,
            **shot,
            "evApplied": ev,
        })
        print(f"{file_name}  yaw={shot['yaw']:>6.1f} pitch={shot['pitch']:>5.1f} ev={ev:+.2f}")

    (out_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"\nwrote {len(shots)} shots + ground_truth.json to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
