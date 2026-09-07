# Top-Down Direction Debugging

Use an annotated top-down render when avatar-relative directions are ambiguous.

For raw Blender interaction previews, render with:

```bash
python scripts/render_raw_blender_interactions.py \
  --motion-pkl /abs/path/assets/neutral_motions/raw_blender_extra_interaction_clips.pkl \
  --metadata /abs/path/assets/neutral_motions/raw_blender_extra_interaction_clips.json \
  --output-dir /abs/path/retarget/output_motion/raw_blender_interaction_previews_extra_topdown \
  --motion <motion_name> \
  --res 720 720 \
  --fps 30 \
  --view-mode top
```

The video overlays `IMAGE UP`, `IMAGE DOWN`, `IMAGE LEFT`, and `IMAGE RIGHT`.
When discussing placement fixes, use those labels directly, for example:
`move bottle IMAGE RIGHT by 0.2` or `move avatar IMAGE UP by 0.15`.

For this top-down debug camera, image-plane offsets are applied as:

```text
IMAGE RIGHT = world +X
IMAGE LEFT  = world -X
IMAGE UP    = world +Y
IMAGE DOWN  = world -Y
```

Prefer image-plane directions for quick visual tuning. Use avatar-local
directions only when the avatar's forward axis has been verified for that
specific motion.
