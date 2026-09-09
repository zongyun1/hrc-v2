# Source provenance

The native root layout is imported from LW-BenchHub commit
`b2bcb2d00edef691f9fcc49039cbf0bcc7464605`.
IsaacLab-Arena is vendored at `third_party/IsaacLab-Arena` from
`c7b70779f103e10d690d1a13863e8d77da7fc782`; its LICENSE.md is preserved.
Exact repository URLs and asset hashes are in `upstream.json`.

The original Lightwheel README is retained as `docs/lightwheel_upstream.md`.
Existing notices and headers are preserved; no new license is assigned to upstream code.
Nested Git metadata and unused Isaac Lab / GR00T submodules are not imported.
LFS payloads are separately supplied and hash-checked by `tools/isaaclab3/assets.py`.

Local changes include the previously validated Lab 3 migration (API locations,
ProxyArray access, XYZW boundaries, native managers), HRC lifecycle integration,
white Panda material restoration, editable package discovery, and installation
against the existing Lab 3 runtime. The original Lab 2 installer is archived in
`history/experiments/lightwheel_original/install.sh`.
The migration helper and original patch records remain in `tools/isaaclab3`.
