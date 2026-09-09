# HARPv2 Genesis versus the local Isaac Lab implementation

Reviewed 2026-09-09. Genesis repository: `coworker-genesis/`, cloned over existing SSH authentication from `EEEEEericKKK/Harp-v2`, commit `6c7c467c9d3754897587fe555c76c96c9fd32cb1`. The Genesis engine submodule is recorded at `0ce793b42c945b6848aad624501b2ca756d3e244` but is not initialized. Body models, TRUMANS weights, RoboCasa assets, and generated outputs are not included in the clone.

Isaac comparison covers the current working files in `isaac_kitchen/` and `robocasa_migration/`, including local saved results. Parent HEAD is `005c782f5a73374e85309da8505121a98f9d14ae`, but the working tree contains substantial pre-existing changes; HEAD alone does not describe this implementation. The root README's legacy Genesis benchmark is not the coworker's new HARPv2 implementation and is not evidence that HARPv2 already supports 44 tasks.

## Recommendation

For developing generated human motion in RoboCasa scenes, I would favor the coworker's Genesis architecture as the starting point. For the already demonstrated handover/navigation interactions and task-outcome auditing, the local Isaac implementation is further along. Neither codebase establishes an overall simulator winner, production benchmark readiness, or reliable human-aware manipulation.

Keep the human-motion contract independent of the simulator. Reuse the coworker's motion schema and generation/export approach in Isaac through a separate adapter if Isaac remains the primary runtime. Before a wholesale migration, compare the same manipulation episodes and human trajectories under the same evaluation criteria.

## Implementation comparison

| Area | Coworker's Genesis | Local Isaac Lab | Assessment |
|---|---|---|---|
| Human motion | TRUMANS export, occupancy input, explicit world-frame conversion, NumPy motion schema | Scripted/legacy Mixamo avatar, walking and gesture controllers | Genesis is better structured for generated motion |
| Human collision representation | 22-link skeleton with 26 bone capsules and a separate skinned surface | Kitchen avatar uses torso and palm collision proxies | Genesis has broader body coverage; neither demonstrates validated human impact dynamics |
| Human–robot coordination | Co-simulation script runs a sinusoidal arm sweep alongside prerecorded walking | Handover sequence, spring-grip receiver, and event-based receiver logic | Isaac has more concrete interaction behavior; neither has general reactive avoidance |
| RoboCasa loading | Direct MJCF import, joint-name initial-state restoration, align/material repairs | MJCF-to-USD conversion plus mesh, collision, material, mass, inertia and joint repairs | Genesis has a shorter project-level import path; Isaac has more explicit audit machinery |
| Manipulation | Recorded EEF poses tracked by arm IK; recorded base/torso targets and gripper timing | Contact-based scripts, navigation control, demonstration following with placement feedback | Different tasks/controllers prevent a fair success-rate ranking |
| Success validation | Replay prints tracking error and object lift; no cabinet-placement predicate | Explicit task predicates, contact checks, hold windows, JSON outcomes; separate upright check for tea | Isaac is stronger here |
| Architecture | Small separated motion/simulation packages and invariant tests | More task-specific scripts and simulator orchestration, with reusable helpers | Genesis is cleaner for extension; some package directories are still placeholders |
| Training and scaling | Batch-shaped schema, but no completed training environment; visible skin batching issue | Standalone prototypes; no complete vectorized RoboCasa training environment | Neither is ready |
| Reproducibility | Engine gitlink pinned; three environments and external assets require setup; simulation outputs absent | Container/Slurm launchers and saved results exist, but installation paths are cluster-specific | Neither is a turnkey portable setup |
| Runtime/render quality | No same-workload measurements or local generated videos in this clone | Local videos/results exist, but no matched Genesis run | Unranked |

## Findings that affect the decision

1. **Genesis's reported pick-and-place is not yet an automatically verified task success.** [replay_demo_ik.py](../coworker-genesis/tools/replay_demo_ik.py) ends by printing EEF error and whether maximum object lift exceeds 3 cm. It does not check cabinet containment, release, or stable final placement and does not fail the process on a failed lift. The progress document reports roughly 55–58 cm lift and millimeter-scale tracking, but these are coworker-reported results; the underlying outputs are excluded from Git. They support a promising demonstration, not a measured benchmark success rate.

2. **Genesis's current proximity signal does not measure full robot–human clearance.** [phase1_cosim_demo.py](../coworker-genesis/tools/phase1_cosim_demo.py) lines 93–103 compare human joint centers only with the robot EEF and base position. This misses robot forearm/elbow geometry and human capsule radii. Contacts are queried against the entire imported kitchen entity, which also includes the robot and floor; that count is not a robot-specific collision count. Use robot-link filtering and surface/capsule distances before reporting safety metrics.

3. **Whole-body collisions do not make the Genesis human dynamically realistic.** [driver.py](../coworker-genesis/hpmm/hpmm_sim/human/driver.py) overwrites qpos with `zero_velocity=True` every simulation step. This is suitable as a prescribed moving obstacle, but resets collision response and does not supply a physically tracked human velocity. Motion frames are selected without interpolation. Contact-force interpretation needs separate validation; a nominal 70 kg mass does not establish realistic impact response. Isaac also uses kinematic human proxies, and its handover uses an explicitly approximated spring grip rather than articulated human finger grasping.

4. **The Genesis batch contract is ahead of its implementation.** `HumanDriver.update_skin()` flattens all environments' link positions/quaternions and indexes them with a single skeleton's permutation (lines 189–191). For multiple environments, this selects the first environment's links and supplies only one skinned mesh's vertices. This is a code-inspection finding, not a GPU reproduction. It needs a per-environment implementation and a test with different poses before treating the human bridge as vectorized. Also, `replay_demo_ik.py` calls `instance.apply(..., strict=False)` and discards the matching report, permitting an incomplete initial-state restoration without a hard failure.

5. **Isaac's successful tea predicate hid an important physical failure.** The saved [comparison.json](../outputs/robocasa_migration/comparisons/serve_tea_demonstration/comparison.json) records `robot_task_success=true` but `upright_task_success=false`: Isaac's final cup tilt was 85.06°, versus 3.22° in the native run. The latest recorded attempt, job 725092, fails both success flags. Local follow-up control changes are documented as not yet simulation-validated. Therefore ServeTea is not solved as an upright-serving task. This also illustrates why Genesis needs more than a lift metric.

6. **Isaac has useful, bounded evidence of other outcomes.** The saved [handover result](../outputs/isaac_kitchen_handover/713442/handover_result.json) records 21.1 cm human carry and 1.33 s final hold using the spring-grip model. [OpenMicrowave job 721878](../outputs/robocasa_migration/probes/721878/summary.json) records approximately 90° opening and task success. [Navigation documentation](../robocasa_migration/MOBILE.md) records a successful fixed-instance navigation run with 7.82 mm final target distance. These do not establish randomized success rates. The Lightwheel kitchen in the handover demo is visual scenery with fixture physics disabled; the separate RoboCasa migration does enable task interaction physics.

## Verification performed

- Genesis CPU suite: **14 passed, 11 skipped**, using an isolated NumPy/pytest environment and `PYTHONPATH=hpmm`. Skips require the absent SMPL-X-derived skeleton/skin assets. This is not a Genesis simulator run.
- Isaac task semantics: **7 tests passed**.
- Isaac USD mobile-physics regression checks: **7 tests passed**, using the existing temporary USD environment.
- Isaac pose-feedback checks: **3 tests passed**.
- Isaac event receiver check: **passed**, covering waiting for capture, transient clearance rejection, sustained clearance, completion and fresh reset.
- Reviewed source, local Isaac JSON results, and coworker documentation. No new GPU simulations, matched performance benchmark, or visual comparison was performed. The tests run here do not validate physical behavior.

## Concrete next comparison

Use the same officially verified demonstration episode in both backends, with identical scene, initial object/robot state, robot embodiment, reference motion and human motion. A practical first shared task is PickPlaceCounterToCabinet, since the Genesis replay already targets it; add the corresponding Isaac task predicate/adapter before comparing. Keep this separate from the harder ServeTea test.

First run without a human to isolate import/control/contact issues. Then run identical human trajectories near the work area. Measure full task success with a final hold, object pose/uprightness where relevant, filtered robot–human contacts, geometry-aware minimum clearance, and object disturbances. Compare cold setup time, warm steps/second and peak memory on the same GPU with rendering off, then under matched camera settings. A pilot of 10 episodes per backend would expose basic robustness issues; it would not prove broad generalization.

My practical choice is to adopt the Genesis motion architecture, retain Isaac's validation discipline and existing successful scenarios, and defer an exclusive engine choice until this shared test exists.
