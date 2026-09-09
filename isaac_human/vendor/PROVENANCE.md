The three NumPy-only modules schema.py, skeleton.py and skin.py are copied
without modification from EEEEEericKKK/Harp-v2 commit
6c7c467c9d3754897587fe555c76c96c9fd32cb1 (2026-09-09 checkout).

Original paths:
- hpmm/hpmm_motion/io/schema.py
- hpmm/hpmm_sim/human/skeleton.py
- hpmm/hpmm_sim/human/skin.py

The new motion.py and adapter.py replace the Genesis driver. No Genesis engine
is imported or installed. Asset paths are always supplied explicitly, so the
upstream modules' original default paths are not used.

The source checkout has no top-level LICENSE file. This copy is for the user's
requested internal coworker integration and does not assign a new license.
SMPL-X-derived body assets and generated motion are not bundled here.
