# Neutral Region Visualization Splits

Goal: render top-down review videos that show the neutral-avatar placement
regions as thin colored blocks just above the table surface. The overlays
should make it easy to verify that candidate regions, blocked regions, and
selected motion footprints are calculated correctly.

## Shared Convention

Use `scripts/render_neutral_task_motion.py --view-mode top --debug-regions`.

Region colors:

- Blue: usable table bounds.
- Red: forbidden tabletop/object regions.
- Orange: candidate neutral motion footprint rejected by the selector.
- Green: candidate neutral motion footprint that is clear but not selected.
- Yellow: selected neutral motion footprint.

When regions overlap, visual priority should be:

- Red above yellow above orange above blue.
- Green is auxiliary and should stay below orange, because it only marks
  unselected clear candidates.

Top-down image directions:

- IMAGE RIGHT = world +X.
- IMAGE LEFT = world -X.
- IMAGE UP = world +Y.
- IMAGE DOWN = world -Y.

Do not use `--debug-direction-markers` for final region videos; those marker
balls are only for direction debugging and make region review noisy.

## Split 1: Shared Overlay + Dump-Bin Wiping

Owner: neut3.

Scope:

- Add the shared debug-region overlay plumbing in
  `envs/tasks/neutral_avatar_table_work.py`.
- Add `--debug-regions` to `scripts/render_neutral_task_motion.py`.
- Verify `dump_bin_neutral` wiping motions:
  - `01_KIT_wipe_table_wiping_the_table01_stageii`
  - `03_KIT_wipe_table_wiping_the_table05_stageii`
- For each wipe motion, render right/back/left forced-side videos with
  `--neutral-sponge-side`.
- Keep neut3 review videos in `data/neut3_latest_videos/`. Do not use the
  crowded shared flat review folder for neut3 outputs.

Expected output names:

- `dump_bin_neutral__01_wipe_table__side_right__region_top_side.mp4`
- `dump_bin_neutral__01_wipe_table__side_back__region_top_side.mp4`
- `dump_bin_neutral__01_wipe_table__side_left__region_top_side.mp4`
- `dump_bin_neutral__03_wipe_table__side_right__region_top_side.mp4`
- `dump_bin_neutral__03_wipe_table__side_back__region_top_side.mp4`
- `dump_bin_neutral__03_wipe_table__side_left__region_top_side.mp4`

Validation:

- Each summary should have `metrics.success == true`.
- Each summary should have `avatar_collision.any_collision == false`.
- The yellow selected footprint should stay on the table and not intersect
  visible task objects.
- The avatar body should initialize outside the selected table side and face
  inward.

## Split 2: Generalize Region Records Across Neutral Tasks

Owner: unassigned.

Scope:

- Audit all neutral tasks for task-specific forbidden regions and table bounds.
- Ensure `_neutral_avatar_forbidden_regions()` only includes tabletop objects
  that should block neutral work for each task.
- Add task-specific side/placement helpers where needed, following the
  dump-bin pattern instead of hard-coding visual-only guesses.
- Do not edit `scripts/render_neutral_task_motion.py` unless Split 1 has a
  clear bug.

Recommended tasks:

- `categorize_neutral`
- `place_bread_in_basket_neutral`
- `place_dual_shoes_neutral`
- `stack_bowls_three_neutral`

## Split 3: Motion Coverage Matrix + Batch Renderer

Owner: unassigned.

Scope:

- Build a small batch script that renders one top-down region video for each
  neutral motion across selected tasks.
- Use one seed by default, with optional `--seeds`.
- Keep output flat for review.
- Avoid duplicating Split 1 overlay code.

Suggested output root:

- `data/neutral_region_debug_matrix/`
- Flat copies in `data/neutral_task_motion_neut4_flat_videos/`.

## Split 4: Review Cleanup + Documentation

Owner: unassigned.

Scope:

- Review generated videos and summaries.
- Remove stale/duplicate review videos older than one hour when asked.
- Update task/motion status docs with which region visualizations pass.
- Capture any remaining bad regions as concrete direction fixes using IMAGE
  directions, not avatar-local directions unless verified.
