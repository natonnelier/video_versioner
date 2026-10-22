# video_versioner

Re-cast a template video with new characters: same video, same motion, performed by the people
in user-uploaded photos (the "Alors on danse" making-of trend, e.g. alorsondanseai.com, which is a
landing page in front of a vhoo.ai template and doesn't name its model).

Scope for now: one or two fixed templates, preprocessed ahead of time, so a request only swaps
the characters.

## Approach

Model: **Wan2.2-Animate-14B** (Alibaba, open weights, https://github.com/Wan-Video/Wan2.2) in
**Replace mode**: the character is regenerated inside the original scene, keeping the background,
camera and lighting.

How the model works, for reference:
- Diffusion Transformer (Wan 2.x image-to-video, 14B) over 3D-VAE video latents.
- Identity: reference photo latents placed alongside the video latents.
- Body motion: 2D pose-skeleton latents, spatially aligned and added to the noisy latents.
- Face/lips: face crops → face encoder (expression without identity) → cross-attention.
- Replace mode: background + person mask given to the model (video inpainting), plus a
  relighting LoRA to match the scene lighting.
- Long clips are generated in chunks conditioned on the previous chunk's last frames.

Alternatives, if Wan-Animate isn't good enough: first-frame edit (Nano Banana, Flux Kontext,
Qwen-Image-Edit) then propagation with Wan VACE / Runway Aleph / Luma Modify; closed APIs such as
Kling motion control, Runway Act-Two, Viggle.

## Code

`pipeline.py`: a single script with two subcommands. Shells out to the Wan2.2 repo scripts and to
ffmpeg/ffprobe. It's only been checked for syntax and argument parsing; it has never run end to
end.

- `prepare` (offline, once per template): reads a template JSON (video path and, per role, a crop
  box `[x, y, w, h]`), crops each role's region, and runs
  `wan/modules/animate/preprocess/preprocess_data.py --replace_flag` on it with a gray placeholder
  reference image. Output goes to `prepared/<template>/`: `template.mp4`, `manifest.json`, and
  `<role>/processed/` containing `src_pose.mp4`, `src_face.mp4`, `src_bg.mp4`, `src_mask.mp4`,
  `src_ref.png`.
- `run` (per request): for each role, copies `<role>/processed/` to a temp dir, overwrites
  `src_ref.png` with the user's photo, runs
  `generate.py --task animate-14B --replace_flag --use_relighting_lora`, then pastes the result back
  at the crop position with ffmpeg (scaled and retimed to the template's fps). Finally adds the
  template's audio back.

Why crops: Wan-Animate's preprocessing targets one main subject, so each pass must see exactly one
person. Crops must contain that person for the whole clip and must not overlap.

`templates/alors_on_danse.json` and the template video don't exist yet. The crop values in the
docstring are placeholders (1920×1080 split down the middle).

## Unverified assumptions

- Script names, CLI flags and output file names (`src_*.mp4`, `src_ref.png`) of the Wan2.2 repo are
  from memory of its README. Check them against the cloned repo.
- In Replace mode, the reference image is only copied through by the preprocessing, and
  `generate.py` resizes/pads it itself. `run` relies on this to reuse the preprocessing output.
  Check: preprocess the same crop with two different photos and diff the outputs. Only
  `src_ref.png` should differ.

## Constraints

- The 14B model needs about 80 GB of GPU memory (A100/H100) on one GPU, or several GPUs with
  `torchrun --dit_fsdp --t5_fsdp --ulysses_size N`. It can't run on the dev Mac; use a cloud GPU
  or a hosted endpoint (fal.ai, Replicate).
- `run --offload` passes `--offload_model True --t5_cpu --convert_model_dtype` to `generate.py`
  for GPUs under 80 GB (e.g. A100 40GB). Slower, and untested; it may also need a lower
  resolution in `prepare`. It won't make a Colab T4 work: 16 GB of VRAM and ~12 GB of system RAM
  are too little for the 14B weights (~28 GB in bf16), and the T4 lacks native bf16. `prepare`
  alone (pose, face, SAM 2) should run on a T4.
- Expect a few minutes of H100 time per 10–15 s clip at 720p, per role.

## Exploration track: hybrid draft + refine

Idea: templates are reused for every request, so move as much work as possible into `prepare`.
Instead of generating the video from pure noise, render a cheap draft of the new character with
the stored template motion, then let a diffusion model refine it (fewer steps, possibly a smaller
model). Not built; the gains below are expectations, not measurements.

Stored per template in `prepare`:
- Per-person tracks and masks (SAM 2, already done).
- Clean background plate with the original people removed (video inpainting: ProPainter, Wan
  VACE or similar remover).
- 3D body motion per frame (body, hands, face expression) and camera (SMPL-X fitting: GVHMR,
  WHAM, SMPLer-X).
- Face motion template (LivePortrait can save a driving video's motion to a reusable file).
- Depth, pose, and the encoded template conditioning.
- The prompt's T5 embedding: the prompt is fixed per template, so `run` never loads the ~11 GB
  T5 encoder.

Per request in `run`:
1. Photo → animatable 3D avatar (e.g. LHM, Alibaba 2025; single image, seconds).
2. Draft: drive the avatar with the stored motion (skinning), render from the stored camera over
   the clean plate. Seconds.
3. Refine (SDEdit-style): VAE-encode the draft, add partial noise (strength ~0.3–0.7), denoise
   conditioned on the photo and the stored conditioning. Expected savings: about half the steps,
   less with step-distillation LoRAs (4–8 steps), and maybe Wan2.2-TI2V-5B instead of 14B. The
   draft also anchors every chunk of a 10–15 s clip, which should reduce identity drift between
   chunks.
4. Optional LivePortrait face pass; put the audio back.

Risks:
1. Licensing (see below) if this becomes commercial.
2. Refinement strength: too low keeps the game-render look, too high means full generation cost
   and drift again. Whether a good middle exists is the main unknown.
3. Hands on objects (keyboard, laptop in the Alors on danse template): noisy hand estimates and
   different arm lengths, so fingers may miss the keys.
4. Single-photo avatars guess the back, sides, loose clothing and hair.
5. Five or six research codebases instead of one.

`experiment_hybrid.py` implements the experiment below for one role and one photo, reusing
`pipeline.py`. Draft: `degraded` (baseline made to look like a game render, an optimistic proxy)
or any video path, e.g. a real avatar render later. Refiner: diffusers `WanVideoToVideoPipeline`
(default Wan2.1-T2V-1.3B), 81-frame chunks at 16 fps, about 480p. It sees only the draft and a
prompt, not the photo, so identity scores are a lower bound. Outputs `baseline.mp4`,
`draft.mp4`, `refine_<s>.mp4`, `side_by_side.mp4` and `report.json` (timings, ArcFace identity
via optional insightface, flicker, seam flicker, PSNR vs baseline). Compiles; never run (needs a
CUDA GPU; diffusers API used from memory).

`experiment_hybrid.ipynb` runs it on Colab: inputs read from `MyDrive/video_versioner/` (scripts,
`templates/`, `photos/`), copied to local disk, results (`prepared/`, `experiments/`) copied back
to Drive so later sessions skip `prepare`. Turns on `--offload` below 70 GB of GPU memory. Needs
an 80 GB GPU (or A100 40 GB with offload, untested) for the baseline; on a T4/L4 only the refine
part runs, from an existing baseline. Never run.

First experiment, before building anything:
1. Baseline: full Wan2.2-Animate with the current pipeline; record quality and GPU time.
2. Hybrid without 3D: a crude draft refined at strengths 0.3 / 0.5 / 0.7 with fewer steps.
3. Compare. If 2 can't approach 1 in less time, the 3D branch won't save it; if it can, build the
   avatar draft.

### Licensing, for future commercial use

SMPL and SMPL-X (Max Planck) are non-commercial research licenses; commercial use needs a paid
license from Meshcapade. STAR and MANO come from the same group under similar terms. Most tools in
the 3D branch output or depend on SMPL-X (GVHMR, WHAM, SMPLer-X, LHM), so swapping the body model
means swapping the estimator and avatar model too. Training datasets (e.g. AMASS) can carry their
own restrictions. Check everything below before relying on it; this is from memory.

Alternatives to SMPL / SMPL-X:
- MHR (Momentum Human Rig, Meta, late 2025): parametric body model released with SAM 3D Body, an
  image → 3D body estimator that outputs MHR. Permissive license as far as I know; that gives an
  estimator and body model together.
- Anny (NAVER LABS, 2025): body model built on MakeHuman, all ages and body shapes, Apache 2.0.
- MakeHuman: open-source human mesh generator, CC0 output; usable as a base mesh with a custom
  rig.
- Game-style skeletons (BVH or FBX rigs, mocap retargeting as in game engines): no body-model
  license at all, but needs a pose estimator that outputs that skeleton, and an avatar rigged to
  it.
- SMPL-X with a paid Meshcapade license: keeps the whole current tool ecosystem.
- Skip 3D entirely: the current Wan-Animate path uses only 2D pose skeletons (no body-model
  license).

Other licenses to check: Wan2.2 is Apache 2.0. LivePortrait's code is MIT, but it relies on
InsightFace models with non-commercial terms. Check LHM, GVHMR and the inpainting models too.

## Next steps

1. Get a GPU machine, clone Wan2.2, download Wan2.2-Animate-14B (incl. `process_checkpoint/`),
   and run the official Replace-mode example as-is to confirm the CLI flags and file names. The
   install and download cells of `experiment_hybrid.ipynb` also work for this on Colab.
2. Verify the reference-image assumption above (diff two preprocessing runs).
3. Get the template video (e.g. `yt-dlp`, rights permitting), pick crop boxes from extracted frames
   (`ffmpeg -i video.mp4 -vf fps=1 frames/%03d.png`), write `templates/alors_on_danse.json`.
4. Run `prepare`, then `run` end to end, and fix what breaks. Watch for seams at crop edges,
   frame-count/fps mismatches between Wan's output and the template, and lighting mismatches.
5. Run the hybrid track's first experiment (baseline vs. crude draft + refine) to decide whether
   the hybrid approach is worth building: `experiment_hybrid.ipynb` on Colab with an 80 GB GPU
   (or `experiment_hybrid.py` on any GPU machine). On the first run, expect to fix the install
   (flash-attn build, torch version) and check the unverified names (Wan2.2 requirement files,
   Hugging Face repo id, diffusers `WanVideoToVideoPipeline` arguments). Try 1.3B and 14B
   refiners, and later a real avatar render via `--draft`.
6. If the two people overlap at any point: replace crops with per-person SAM 2 masks (click- or
   box-prompted, tracked through the clip) fed into the preprocessing.
7. Optional: auto-generate crop boxes with a person detector + tracker (ultralytics YOLO +
   ByteTrack), taking the union of each track's boxes over the clip.
8. Keep the model loaded: replace the per-role `generate.py` subprocess (which reloads 14B weights
   every call) with a long-running worker that loads the pipeline once and serves requests. Run the
   two roles in parallel on two GPUs, since the passes are independent.
9. Then a thin product layer: upload two photos → queue job → return video. Or swap step 2 of the
   pipeline for a hosted Wan-Animate API to skip owning GPUs.
