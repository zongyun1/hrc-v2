# Task matrix status (GPU, video stride 10)

Baseline completed on 2026-07-11 from commit `75abff6` using:

```bash
GENESIS_BACKEND=gpu .venv-genesis-latest/bin/python scripts/run_task_matrix.py \
  --episodes 1 --video-stride 10 \
  --save-root outputs/task_matrix_stride10
```

The serial run completed all 53 registered tasks in about 8 hours 22 minutes.
Results: **21 passed, 31 failed, and 1 timed out** (32 tasks requiring repair).
The raw, ignored run report is at `outputs/task_matrix_stride10/summary.json`.

## Results

| # | Task | Status | Time (s) |
|---:|---|---|---:|
| 1 | `take_from_human_easy` | Failed | 303.7 |
| 2 | `deliver_to_human_easy` | Passed | 149.0 |
| 3 | `take_from_human_safety` | Failed | 321.3 |
| 4 | `stamp_documents` | Passed | 1436.3 |
| 5 | `pour_water` | Passed | 404.7 |
| 6 | `frying_with_robot_pour` | Failed | 158.7 |
| 7 | `oil_bottle_recovery` | Failed | 172.1 |
| 8 | `open_microwave` | Passed | 862.9 |
| 9 | `categorize_cooperative` | Failed | 1261.4 |
| 10 | `categorize_neutral` | Failed | 1384.3 |
| 11 | `put_object_cabinet` | Failed | 109.7 |
| 12 | `stack_bowls_three` | Passed | 91.9 |
| 13 | `place_bread_in_basket` | Passed | 84.0 |
| 14 | `dump_bin` | Failed | 90.3 |
| 15 | `dump_bin_xarm_calibrated` | Failed | 99.2 |
| 16 | `place_burger_fries` | Failed | 98.2 |
| 17 | `place_dual_shoes` | Passed | 103.9 |
| 18 | `place_food_in_skillet` | Failed | 97.2 |
| 19 | `blocks_ranking_rgb` | Passed | 142.9 |
| 20 | `blocks_ranking_size` | Passed | 147.7 |
| 21 | `put_object_cabinet_assist` | Failed | 251.3 |
| 22 | `put_object_cabinet_interrupt` | Failed | 278.8 |
| 23 | `put_object_cabinet_neutral` | Failed | 1569.9 |
| 24 | `stack_bowls_three_assist` | Passed | 350.9 |
| 25 | `stack_bowls_three_interrupt` | Failed | 109.0 |
| 26 | `stack_bowls_three_neutral` | Failed | 1382.2 |
| 27 | `place_bread_in_basket_assist` | Passed | 1010.3 |
| 28 | `place_bread_in_basket_interrupt` | Passed | 103.9 |
| 29 | `place_bread_in_basket_neutral` | Passed | 1435.0 |
| 30 | `dump_bin_assist` | Failed | 78.3 |
| 31 | `dump_bin_interrupt` | Failed | 93.7 |
| 32 | `dump_bin_xarm_calibrated_assist` | Failed | 65.2 |
| 33 | `dump_bin_xarm_calibrated_interrupt` | Failed | 114.1 |
| 34 | `dump_bin_neutral` | Timed out | 1800.4 |
| 35 | `place_burger_fries_assist` | Failed | 828.7 |
| 36 | `place_burger_fries_interrupt` | Passed | 183.0 |
| 37 | `place_burger_fries_neutral` | Failed | 1322.2 |
| 38 | `place_dual_shoes_assist` | Passed | 253.4 |
| 39 | `place_dual_shoes_interrupt` | Passed | 127.4 |
| 40 | `place_dual_shoes_neutral` | Failed | 1249.5 |
| 41 | `place_food_in_skillet_assist` | Failed | 907.5 |
| 42 | `place_food_in_skillet_interrupt` | Failed | 161.4 |
| 43 | `place_food_in_skillet_neutral` | Failed | 1225.0 |
| 44 | `blocks_ranking_rgb_assist` | Passed | 892.3 |
| 45 | `blocks_ranking_rgb_interrupt` | Passed | 308.7 |
| 46 | `blocks_ranking_rgb_neutral` | Failed | 1737.5 |
| 47 | `blocks_ranking_size_assist` | Passed | 977.7 |
| 48 | `blocks_ranking_size_interrupt` | Passed | 249.1 |
| 49 | `blocks_ranking_size_neutral` | Failed | 1611.4 |
| 50 | `suitcase_organize_interrupt` | Failed | 163.7 |
| 51 | `factory_inspect_pack` | Failed | 994.1 |
| 52 | `factory_kit_packing` | Failed | 397.4 |
| 53 | `factory_line_feeding` | Passed | 357.8 |

## Repair workflow

Failures are repaired and verified one task at a time. Every verification run
must explicitly set `GENESIS_BACKEND=gpu`; CPU results do not close an item.
After a task passes seed 0, its status and verification evidence are recorded
here before moving to the next failure.

## Repairs

| Task | State | GPU verification | Notes |
|---|---|---|---|
| `take_from_human_easy` | Fixed | Seed 0 passed in 217.7 s | Partial last-mile descent reduced final lateral error from 0.138 m to 0.016 m without entering the unstable low-IK region. |

Current repair target: `take_from_human_safety`.
