# Mixamo Motion → Vico Motion Conversion

End-to-end guide for converting a Mixamo `.fbx` animation into the motion
format consumed by `envs.avatar.controller.AvatarController`, and rendering
the result in Genesis as a video.

## Output format

Generated motion files are dicts keyed by motion name. Each value has:

| key     | shape             | description                                |
|---------|-------------------|--------------------------------------------|
| `trans` | `(T, 3)`          | root translation (m), Y-up                 |
| `rot`   | `(T, 4)`          | root rotation quaternion (w, x, y, z)      |
| `joint` | `(T, 64, 4)`      | per-joint quaternions (w, x, y, z)         |
| `mat`   | `(T, 65, 3, 3)`   | per-joint rest-pose basis matrices         |

`T` is the animation length in frames. The 65-bone hierarchy matches the
Mixamo skeleton (Hips → Spine → ... → toes + fingers); see
`retarget/export_motion_from_blender.py` for the full bone order.

`AvatarController` loads `assets/avatars/motions/motion.pkl` for the baked-in
motions (idle, walk, stand) and merges generated motions from
`assets/avatars/motions/generated_motions.pkl` by default. Use
`retarget/output_motion/` only as a scratch conversion directory, then merge
accepted motions into the assets pickle.

## Two conversion paths

### Path A — `mixamo_to_vico.py` + `fbx-extract` (recommended, no Blender)

1. Drop `.fbx` files into `retarget/raw_motions/` (any Mixamo Ch29 rig
   works).
2. Run the `fbx-extract` binary once per file. It writes a set of
   `*_skel_local.txt`, `*_hierarchy.txt`, `*_static_transforms.txt`,
   `*_binding_pose_local.txt`, mesh `.obj`, and textures alongside the
   `.fbx`.
3. Run `retarget/mixamo_to_vico.py`, which walks `--extract_root`, reads
   the txt files, applies the Mixamo→Vico coordinate fix (converts each
   joint's Euler track through `coord_mat @ R @ coord_mat_inv`), and
   writes `retarget/output_motion/motion.pkl`.

The script auto-runs the binary when `*_skel_local.txt` is missing, so
step 2 usually happens implicitly the first time.

#### Example — convert one motion only

```bash
# Pre-extract raw_motions/ contains the .fbx and, optionally, extracted
# txt files. To avoid reprocessing every .fbx, stage just the one you
# want:
mkdir -p retarget/shove_only
for f in "Shove Reaction.fbx" \
         "Shove Reaction_skel_local.txt" \
         "Shove Reaction_static_transforms.txt" \
         "Shove Reaction_hierarchy.txt"; do
  ln -sf "$PWD/retarget/raw_motions/$f" "retarget/shove_only/$f"
done

# If the txt files don't exist yet, run fbx-extract once:
/work/pi_chuangg_umass_edu/qinhongzhou/fbx-extract/build/fbx-extract \
    "retarget/raw_motions/Shove Reaction.fbx" \
    "retarget/raw_motions"

# Convert
/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
    retarget/mixamo_to_vico.py \
    -e retarget/shove_only \
    -o retarget/output_motion
```

Output: `retarget/output_motion/motion.pkl` with key `"Shove_Reaction"` or
the normalized key produced by the converter.
After visual validation, merge the key into
`assets/avatars/motions/generated_motions.pkl`; runtime tasks should not depend
on `retarget/output_motion/`.

#### Incremental merge

To add a new motion to an existing `motion.pkl` without recomputing the
others, pass `--incremental --motion-name NAME --motion-frag START,END,STEP`.
The script loads the existing pickle, adds just that motion, and refuses
to overwrite an existing key.

### Path B — Blender (`import_and_export_fbx.py`)

Use when the FBX has a non-standard skeleton or the `fbx-extract` binary
isn't available. Requires Blender ≥ 4.2 on PATH:

```bash
BLENDER=/work/pi_chuangg_umass_edu/qinhongzhou/blender/blender-4.2.1-linux-x64/blender

$BLENDER -b -P retarget/import_and_export_fbx.py -- \
    --fbx  retarget/raw_motions/Shove\ Reaction.fbx \
    --name "Shove Reaction" \
    --output retarget/output_motion/motion.pkl \
    --merge  retarget/output_motion/motion.pkl
```

Internals: `import_and_export_fbx.py` imports the FBX into an empty
Blender scene, finds the armature, and calls
`export_motion_from_blender.export_motion_from_blender()`. That function
reads bone transforms directly from Blender's pose API, converts
Blender's Z-up → Y-up, applies the same `coord_mat` basis fix as path A,
and writes the same `motion.pkl` format.

## Rendering the converted motion

`scripts/play_motion.py` loads the avatar, merges in
`assets/avatars/motions/generated_motions.pkl` by default, and records an MP4:

```bash
/work/pi_chuangg_umass_edu/qinhongzhou/conda/robotwin/bin/python \
    scripts/play_motion.py --motion Shove_Reaction -o shove_reaction.mp4
```

Rendering needs a GPU for EGL. On the cluster, submit via SLURM:

```bash
sbatch scripts/run_play_motion.sh            # defaults to Shove_Reaction
MOTION="My Motion" OUTPUT=my.mp4 sbatch scripts/run_play_motion.sh
```

The job reserves one GPU on `gpu-preempt`, sets `PYOPENGL_PLATFORM=egl`,
and writes the video to the repo root. Logs go to
`scripts/run_play_motion.log`.

## Quick sanity checks on `motion.pkl`

```python
import pickle
d = pickle.load(open("assets/avatars/motions/generated_motions.pkl", "rb"))
for k, v in d.items():
    print(k, v["trans"].shape, v["rot"].shape, v["joint"].shape, v["mat"].shape)
```

Expected: `(T, 3) (T, 4) (T, 64, 4) (T, 65, 3, 3)` for some `T`. If
`joint.shape[1] != 64` or `mat.shape[1] != 65`, the skeleton does not
match the Mixamo hierarchy in
`retarget/export_motion_from_blender.py:32-54` and the avatar will skip
or misplace bones.

## Known pitfalls

- The motion name is usually the normalized filename stem. Check keys before
  rendering and quote names that contain spaces.
- `AvatarController` requires `idle`, `stand`, and `walk` keys to exist
  in the baked `assets/avatars/motions/motion.pkl` at construction.
  Generated motions only need to define their own key.
- If EGL fails with `EGL_BAD_MATCH`, the node has no GPU — run through
  SLURM.
