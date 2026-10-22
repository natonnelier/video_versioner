"""Re-cast a template video with new characters using Wan2.2-Animate-14B (Replace mode).

Two commands:

  prepare  (once per template, offline)
    For each role: crop the role's region out of the template (Wan-Animate's
    preprocessing targets one main subject, so each crop must contain one person),
    then run the official preprocessing: pose, face crops, background, person mask.
    Everything lands in a prepared directory with a manifest.

  run  (per request)
    For each role: copy the prepared inputs, drop the user's photo in as the
    reference image, generate with Wan2.2-Animate-14B, paste the result back
    into the template. Finally, put the template's original audio back.

Requires a clone of https://github.com/Wan-Video/Wan2.2 with its dependencies,
the Wan2.2-Animate-14B checkpoint (incl. process_checkpoint/), ffmpeg and ffprobe.

Usage:
  python pipeline.py prepare --template templates/alors_on_danse.json \
      --wan-repo ~/Wan2.2 --ckpt ~/Wan2.2-Animate-14B --out prepared/alors_on_danse

  python pipeline.py run --prepared prepared/alors_on_danse \
      --role keyboard=alice.jpg --role laptop=bob.jpg \
      --wan-repo ~/Wan2.2 --ckpt ~/Wan2.2-Animate-14B --out out.mp4 [--offload]

Template file (paths relative to the current directory):
  {
    "video": "templates/alors_on_danse.mp4",
    "roles": {
      "keyboard": {"crop": [0, 0, 960, 1080]},
      "laptop":   {"crop": [960, 0, 960, 1080]}
    }
  }
Crop boxes are [x, y, width, height] in pixels (even values), must contain one
person for the whole clip, and must not overlap each other.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Files the preprocessing writes and generate.py reads in Replace mode.
# Only the reference image depends on the user's photo; the rest depends on the template alone.
REF_FILE = "src_ref.png"
TEMPLATE_FILES = ["src_pose.mp4", "src_face.mp4", "src_bg.mp4", "src_mask.mp4"]


def run_cmd(cmd: list, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)), file=sys.stderr)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def frame_rate(video: Path) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()  # e.g. "30/1"


def crop(video: Path, box: list[int], out: Path) -> Path:
    x, y, w, h = box
    run_cmd(["ffmpeg", "-y", "-i", video, "-vf", f"crop={w}:{h}:{x}:{y}", "-an", out])
    return out


def placeholder_image(out: Path) -> Path:
    # The preprocessing script requires a reference image; in Replace mode it is only copied through.
    run_cmd(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=gray:s=512x512", "-frames:v", 1, out])
    return out


def preprocess(wan_repo: Path, ckpt: Path, video: Path, image: Path, out_dir: Path) -> Path:
    run_cmd([
        sys.executable, "wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path", ckpt / "process_checkpoint",
        "--video_path", video.resolve(),
        "--refer_path", image.resolve(),
        "--save_path", out_dir.resolve(),
        "--resolution_area", 1280, 720,
        "--iterations", 3,
        "--k", 7,
        "--w_len", 1,
        "--h_len", 1,
        "--replace_flag",
    ], cwd=wan_repo)
    missing = [f for f in TEMPLATE_FILES if not (out_dir / f).exists()]
    if missing:
        sys.exit(f"preprocessing did not produce {missing} in {out_dir}")
    return out_dir


def generate(wan_repo: Path, ckpt: Path, processed: Path, out: Path, offload: bool = False) -> Path:
    # Single GPU. For multi-GPU, prefix with torchrun and add --dit_fsdp --t5_fsdp --ulysses_size N.
    # offload trades speed for memory: moves models to CPU between steps, keeps T5 on CPU,
    # and casts weights to the compute dtype. For GPUs under 80 GB; untested on 40 GB.
    offload_flags = ["--offload_model", "True", "--t5_cpu", "--convert_model_dtype"] if offload else []
    run_cmd([
        sys.executable, "generate.py",
        "--task", "animate-14B",
        "--ckpt_dir", ckpt,
        "--src_root_path", processed.resolve(),
        "--refert_num", 1,
        "--replace_flag",
        "--use_relighting_lora",
        "--save_file", out.resolve(),
        *offload_flags,
    ], cwd=wan_repo)
    return out


def paste(base: Path, generated: Path, box: list[int], out: Path) -> Path:
    # Wan renders at its own resolution and frame rate: scale and retime to the base before overlaying.
    x, y, w, h = box
    fps = frame_rate(base)
    run_cmd([
        "ffmpeg", "-y", "-i", base, "-i", generated,
        "-filter_complex", f"[1:v]scale={w}:{h},fps={fps}[g];[0:v][g]overlay={x}:{y}:shortest=1",
        "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out,
    ])
    return out


def mux_audio(video: Path, template: Path, out: Path) -> Path:
    run_cmd([
        "ffmpeg", "-y", "-i", video, "-i", template,
        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", out,
    ])
    return out


def prepare(template: Path, wan_repo: Path, ckpt: Path, out: Path) -> Path:
    cfg = json.loads(template.read_text())
    out.mkdir(parents=True, exist_ok=True)
    video = out / "template.mp4"
    shutil.copy(cfg["video"], video)
    placeholder = placeholder_image(out / "placeholder.png")

    roles = {}
    for name, role in cfg["roles"].items():
        role_dir = out / name
        role_dir.mkdir(exist_ok=True)
        clip = crop(video, role["crop"], role_dir / "crop.mp4")
        preprocess(wan_repo, ckpt, clip, placeholder, role_dir / "processed")
        roles[name] = {"crop": role["crop"], "processed": f"{name}/processed"}

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps({"video": "template.mp4", "roles": roles}, indent=2))
    return manifest


def recast(prepared: Path, images: dict[str, Path], wan_repo: Path, ckpt: Path, out: Path,
           offload: bool = False) -> Path:
    manifest = json.loads((prepared / "manifest.json").read_text())
    missing = set(manifest["roles"]) - set(images)
    unknown = set(images) - set(manifest["roles"])
    if missing or unknown:
        sys.exit(f"missing roles: {sorted(missing)}; unknown roles: {sorted(unknown)}")

    template = prepared / manifest["video"]
    with tempfile.TemporaryDirectory(prefix="recast-") as tmp:
        work = Path(tmp)
        current = template
        for name, role in manifest["roles"].items():
            # Work on a copy so the prepared inputs stay untouched between requests.
            processed = shutil.copytree(prepared / role["processed"], work / name / "processed")
            shutil.copy(images[name], processed / REF_FILE)
            generated = generate(wan_repo, ckpt, processed, work / name / "generated.mp4", offload)
            current = paste(current, generated, role["crop"], work / name / "composite.mp4")
        return mux_audio(current, template, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="preprocess a template once")
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("run", help="re-cast a prepared template with new photos")
    r.add_argument("--prepared", type=Path, required=True)
    r.add_argument("--role", action="append", default=[], metavar="NAME=IMAGE")
    r.add_argument("--out", type=Path, default=Path("out.mp4"))
    r.add_argument("--offload", action="store_true",
                   help="offload models to CPU to fit GPUs under 80 GB (slower)")

    for s in (p, r):
        s.add_argument("--wan-repo", type=Path, required=True)
        s.add_argument("--ckpt", type=Path, required=True)

    args = parser.parse_args()
    wan_repo, ckpt = args.wan_repo.expanduser(), args.ckpt.expanduser().resolve()

    if args.command == "prepare":
        print(prepare(args.template, wan_repo, ckpt, args.out))
    else:
        images = {name: Path(path) for name, path in (a.split("=", 1) for a in args.role)}
        print(recast(args.prepared, images, wan_repo, ckpt, args.out, args.offload))


if __name__ == "__main__":
    main()
