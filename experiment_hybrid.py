"""Hybrid experiment: can "cheap draft + diffusion refine" compete with full Wan2.2-Animate?

For one role of a prepared template (see pipeline.py) and one photo:
  1. baseline  full Wan2.2-Animate-14B generation of the role's crop (what `pipeline.py run` does
               for one role, without pasting back)
  2. draft     a stand-in for the future avatar render. Default: the baseline degraded to look
               like a game render, so it has correct motion and identity but poor rendering.
               That's an optimistic draft. Or pass your own video, e.g. a real avatar render.
  3. refine    SDEdit-style: noise the draft partially and denoise it with Wan text-to-video
               (diffusers' WanVideoToVideoPipeline), at several strengths. With strength s and
               N steps, only about s * N steps actually run.
  4. compare   timings, identity similarity to the photo (if insightface is installed), flicker,
               flicker at refine chunk seams, PSNR to the baseline, and a side-by-side video.

The refiner only sees the draft and a prompt, not the photo, so identity has to survive from the
draft alone. A production refiner would also be conditioned on the photo; treat identity scores
as a lower bound.

Requires what pipeline.py requires, plus a CUDA GPU and: torch, diffusers, transformers,
accelerate, ftfy, opencv-python, numpy, imageio-ffmpeg. Optional: insightface and
onnxruntime-gpu, for identity scores.

Usage:
  python experiment_hybrid.py --prepared prepared/alors_on_danse --role keyboard --photo alice.jpg \
      --wan-repo ~/Wan2.2 --ckpt ~/Wan2.2-Animate-14B --out experiments/alice_keyboard

  # reuse that baseline, try another draft or refiner without regenerating it:
  python experiment_hybrid.py --prepared prepared/alors_on_danse --role keyboard --photo alice.jpg \
      --baseline experiments/alice_keyboard/baseline.mp4 --draft avatar_render.mp4 \
      --refine-model Wan-AI/Wan2.1-T2V-14B-Diffusers --offload --out experiments/alice_keyboard_14b
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanVideoToVideoPipeline
from diffusers.utils import export_to_video, load_video
from PIL import Image

import pipeline

FPS = 16  # Wan text-to-video is trained at 16 fps
CHUNK = 81  # frames per refine pass (about 5 s), the clip length Wan is trained on
DEFAULT_PROMPT = (
    "A person performing in a room, realistic photo, natural skin texture, detailed clothing, "
    "natural lighting, sharp focus"
)
NEGATIVE_PROMPT = (
    "cartoon, 3d render, cgi, video game, plastic skin, flat shading, blurry, low quality, "
    "distorted face, deformed hands, extra fingers"
)


def run_baseline(prepared: Path, role: str, photo: Path, wan_repo: Path, ckpt: Path, out: Path,
                 offload: bool) -> float:
    manifest = json.loads((prepared / "manifest.json").read_text())
    if role not in manifest["roles"]:
        sys.exit(f"unknown role {role!r}; template roles: {sorted(manifest['roles'])}")
    with tempfile.TemporaryDirectory(prefix="baseline-") as tmp:
        processed = shutil.copytree(prepared / manifest["roles"][role]["processed"], Path(tmp) / "processed")
        shutil.copy(photo, processed / pipeline.REF_FILE)
        start = time.perf_counter()
        pipeline.generate(wan_repo, ckpt, processed, out, offload)
        return time.perf_counter() - start  # includes loading the 14B model


def target_size(width: int, height: int, area: int = 832 * 480) -> tuple[int, int]:
    # Keep the aspect ratio at about 480p, both sides multiples of 16 as Wan requires.
    scale = (area / (width * height)) ** 0.5
    return round(width * scale / 16) * 16, round(height * scale / 16) * 16


def load_frames(video: Path, work: Path, tag: str, size: tuple[int, int] | None = None) -> list[Image.Image]:
    resampled = work / f"{tag}_{FPS}fps.mp4"
    pipeline.run_cmd(["ffmpeg", "-y", "-i", video, "-vf", f"fps={FPS}", "-an", resampled])
    frames = load_video(str(resampled))
    size = size or target_size(*frames[0].size)
    return [f.convert("RGB").resize(size, Image.LANCZOS) for f in frames]


def render_like(frame: Image.Image) -> Image.Image:
    # Stand-in for an avatar render: lose fine detail, flatten shading, quantize colors.
    img = np.asarray(frame)
    h, w = img.shape[:2]
    img = cv2.resize(cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA), (w, h),
                     interpolation=cv2.INTER_LINEAR)
    for _ in range(2):
        img = cv2.bilateralFilter(img, 9, 75, 75)
    img = (img // 32) * 32 + 16
    return Image.fromarray(img)


def load_refiner(model_id: str, flow_shift: float, offload: bool) -> WanVideoToVideoPipeline:
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanVideoToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def refine(pipe: WanVideoToVideoPipeline, frames: list[Image.Image], strength: float, steps: int,
           prompt: str, seed: int) -> list[Image.Image]:
    width, height = frames[0].size
    out = []
    for start in range(0, len(frames), CHUNK):
        chunk = frames[start:start + CHUNK]
        # Wan needs 4k+1 frames: pad with the last frame, drop the padding afterwards.
        padded = chunk + [chunk[-1]] * ((1 - len(chunk)) % 4)
        result = pipe(
            video=padded, prompt=prompt, negative_prompt=NEGATIVE_PROMPT,
            height=height, width=width, strength=strength, num_inference_steps=steps,
            guidance_scale=5.0, generator=torch.Generator("cuda").manual_seed(seed),
            output_type="pil",
        ).frames[0]
        out.extend(result[:len(chunk)])
    return out


def flicker(frames: list[Image.Image]) -> np.ndarray:
    # Mean absolute change between consecutive frames, in 0-255 gray levels.
    gray = [np.asarray(f.convert("L"), dtype=np.float32) for f in frames]
    return np.array([np.abs(b - a).mean() for a, b in zip(gray, gray[1:])])


def psnr(frames: list[Image.Image], reference: list[Image.Image]) -> float:
    mse = np.mean([np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2)
                   for a, b in zip(frames, reference)])
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-8)))


def identity_scorer(photo: Path):
    """Returns frames -> (mean cosine similarity to the photo's face, face detection rate), or None."""
    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("insightface not installed: skipping identity scores", file=sys.stderr)
        return None
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    def embed(img: Image.Image):
        faces = app.get(cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR))
        if not faces:
            return None
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.normed_embedding

    reference = embed(Image.open(photo))
    if reference is None:
        print(f"no face found in {photo}: skipping identity scores", file=sys.stderr)
        return None

    def score(frames: list[Image.Image], every: int = 8) -> tuple[float | None, float]:
        sampled = frames[::every]
        sims = [float(e @ reference) for e in map(embed, sampled) if e is not None]
        return (float(np.mean(sims)) if sims else None), len(sims) / len(sampled)

    return score


def side_by_side(variants: dict[str, list[Image.Image]], out: Path) -> None:
    rows = []
    for i in range(min(len(v) for v in variants.values())):
        panels = []
        for name, frames in variants.items():
            panel = np.asarray(frames[i]).copy()
            for color, thickness in (((0, 0, 0), 4), ((255, 255, 255), 2)):
                cv2.putText(panel, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness, cv2.LINE_AA)
            panels.append(panel)
        rows.append(Image.fromarray(np.hstack(panels)))
    export_to_video(rows, str(out), fps=FPS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="existing baseline video; skips the baseline run")
    parser.add_argument("--wan-repo", type=Path, help="needed unless --baseline is given")
    parser.add_argument("--ckpt", type=Path, help="needed unless --baseline is given")
    parser.add_argument("--draft", default="degraded", help="'degraded' (from the baseline) or a video path")
    parser.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--steps", type=int, default=40, help="full schedule; about strength * steps run")
    parser.add_argument("--refine-model", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--flow-shift", type=float, default=3.0, help="3.0 for 480p, 5.0 for 720p")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offload", action="store_true", help="offload models to CPU (baseline and refiner)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    report = {"role": args.role, "photo": str(args.photo), "draft": args.draft,
              "refine_model": args.refine_model, "steps": args.steps, "seconds": {}}

    baseline_video = args.baseline
    if baseline_video is None:
        if not (args.wan_repo and args.ckpt):
            sys.exit("--wan-repo and --ckpt are required to run the baseline")
        baseline_video = args.out / "baseline.mp4"
        report["seconds"]["baseline (incl. model load)"] = run_baseline(
            args.prepared, args.role, args.photo, args.wan_repo.expanduser(),
            args.ckpt.expanduser().resolve(), baseline_video, args.offload)

    baseline = load_frames(baseline_video, work, "baseline")
    if args.draft == "degraded":
        draft = [render_like(f) for f in baseline]
    else:
        draft = load_frames(Path(args.draft), work, "draft", baseline[0].size)
    n = min(len(baseline), len(draft))
    baseline, draft = baseline[:n], draft[:n]
    export_to_video(draft, str(args.out / "draft.mp4"), fps=FPS)
    variants = {"baseline": baseline, "draft": draft}

    start = time.perf_counter()
    pipe = load_refiner(args.refine_model, args.flow_shift, args.offload)
    report["seconds"]["refiner load"] = time.perf_counter() - start
    for strength in args.strengths:
        name = f"refine {strength}"
        torch.cuda.synchronize()
        start = time.perf_counter()
        frames = refine(pipe, draft, strength, args.steps, args.prompt, args.seed)
        torch.cuda.synchronize()
        report["seconds"][name] = time.perf_counter() - start
        export_to_video(frames, str(args.out / f"refine_{strength}.mp4"), fps=FPS)
        variants[name] = frames
    del pipe
    torch.cuda.empty_cache()

    score_identity = identity_scorer(args.photo)
    seams = range(CHUNK, n, CHUNK)
    report["metrics"] = {}
    for name, frames in variants.items():
        diffs = flicker(frames)
        identity, face_rate = score_identity(frames) if score_identity else (None, None)
        report["metrics"][name] = {
            "identity": identity,
            "face_rate": face_rate,
            "flicker": float(diffs.mean()),
            "seam_flicker": float(np.mean([diffs[i - 1] for i in seams])) if name.startswith("refine") and seams else None,
            "psnr_vs_baseline": psnr(frames, baseline) if name != "baseline" else None,
        }
    side_by_side(variants, args.out / "side_by_side.mp4")
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    def fmt(value, spec):
        return "-" if value is None else format(value, spec)

    print("\n| variant | identity | face rate | flicker | seam flicker | PSNR vs baseline |")
    print("|---|---|---|---|---|---|")
    for name, m in report["metrics"].items():
        print(f"| {name} | {fmt(m['identity'], '.3f')} | {fmt(m['face_rate'], '.0%')} | {m['flicker']:.2f} "
              f"| {fmt(m['seam_flicker'], '.2f')} | {fmt(m['psnr_vs_baseline'], '.1f')} |")
    print("\n| step | seconds |\n|---|---|")
    for name, seconds in report["seconds"].items():
        print(f"| {name} | {seconds:.0f} |")
    print(f"\nWatch {args.out / 'side_by_side.mp4'}: the numbers don't capture realism.")


if __name__ == "__main__":
    main()
