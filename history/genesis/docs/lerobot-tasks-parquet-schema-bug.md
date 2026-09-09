# Bug report: `meta/tasks.parquet` schema mismatch in `*_target` lerobot datasets

**Summary:** the `meta/tasks.parquet` file in `zhouqh/hrbench/lerobot_datasets/genesis_hr_bench_lerobot_qpos_target` ships with the wrong pandas layout for lerobot v0.5.2 / v3.0 spec. Task strings are in a regular **column** instead of being the **pandas index**, which causes lerobot's task lookup (`self.tasks.loc[task_string]`) to silently return the integer row index instead of the task string. Training then crashes on the first batch with confusing errors (different errors per policy family). Affects only the `_target` variant; the three `_v30` variants are fine.

**Date:** 2026-05-26
**Severity:** lerobot training blocker (pi0 + pi0.5 both fail first-batch)
**Affected dataset:** `huggingface.co/datasets/zhouqh/hrbench` @ `lerobot_datasets/genesis_hr_bench_lerobot_qpos_target/meta/tasks.parquet`

## TL;DR

| | NEW `qpos_target` (broken) | OLD `qpos_v30` (works) |
|---|---|---|
| `df.index.name` | `None` | `task` |
| `list(df.columns)` | `['task', 'task_index']` | `['task_index']` |
| First row | `task='place the red, green, ...'`, `task_index=0` | `task_index=0` (index label is the task string) |
| `LeRobotDataset(...)[0]['task']` | `0` (an `int`) | `'place the red, green, ...'` (a `str`) |

## Fix (one-line pandas op)

```python
import pandas as pd
p = ".../meta/tasks.parquet"
df = pd.read_parquet(p)
df = df.set_index("task")    # move 'task' col → pandas index named 'task'
df.to_parquet(p)
```

The corrected file should match lerobot's `dataset_metadata.py:389` construction pattern:

```python
self.tasks = pd.DataFrame({"task_index": task_indices},
                          index=pd.Index(tasks, name="task"))
```

## How it surfaces in training

Different policies fail differently because the `task` field ends up the wrong type. Both crash on the first batch after `Start offline training`:

| Policy | Where it crashes | Error | Root cause |
|---|---|---|---|
| pi0 | `lerobot/processor/tokenizer_processor.py:185` | `ValueError: Task cannot be None` | tokenizer's `get_task()` returns `None` because the lookup fell back to the int row index, which isn't `str`/`list[str]` |
| pi0.5 | `lerobot/policies/pi05/processor_pi05.py:81` | `AttributeError: 'Tensor' object has no attribute 'strip'` | the int gets stacked into a Tensor by the batch collator, then `task.strip()` fails on the Tensor |

Neither error message points at the actual root cause (the parquet schema). To diagnose, sample one item from the dataset and check the type of `item['task']`:

```python
import os
os.environ["HF_LEROBOT_HOME"] = "/path/to/your/lerobot/cache"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("genesis-hr-bench/genesis_hr_bench_lerobot_qpos_target")
item = ds[0]
print(type(item["task"]), repr(item["task"]))    # broken: <class 'int'> 0
                                                 # fixed:  <class 'str'> 'place the red, ...'
```

## What lerobot does internally

`baseline/lerobot/src/lerobot/datasets/dataset_metadata.py` (line 367) does:

```python
return int(self.tasks.loc[task].task_index)
```

This expects `self.tasks` to be a DataFrame indexed by task string. With the broken layout, `self.tasks.loc[<task_string>]` doesn't find a matching index label and the chain ends up returning an int (the row position) instead of raising a clear KeyError. Result: silent wrong-type propagation through the data loader, with the crash deferred until a policy-specific processor tries to use the value.

## Action items for the data exporter

1. **Confirm the layout intent.** Was the column layout (`['task', 'task_index']`) deliberate, or did the export pipeline accidentally drop the index assignment?

2. **Re-export `genesis_hr_bench_lerobot_qpos_target` with the correct index layout.** Use `df.set_index("task").to_parquet(...)` (or pass `index=pd.Index(tasks, name="task")` when constructing the DataFrame). Once re-uploaded, future collaborators won't need to apply the local fix on download.

3. **Audit the export pipeline against the other variants.** Of the 4 dirs at `zhouqh/hrbench/lerobot_datasets/`:

   | Dataset | Layout |
   |---|---|
   | `genesis_hr_bench_lerobot_v30` (ee) | ✓ index-based (correct) |
   | `genesis_hr_bench_lerobot_qpos_v30` (qpos) | ✓ index-based (correct) |
   | `genesis_hr_bench_lerobot_qpos_abs_v30` (qpos_abs) | ✓ index-based (correct, presumed — match `v30` family) |
   | `genesis_hr_bench_lerobot_qpos_target` (qpos, new curation) | ✗ **column-based — broken** |

   Only the `_target` variant diverged. The three `_v30` variants are fine. Likely a different code path was used to export `_target`; that path is what needs fixing.

## Action items for lerobot upstream (separate, optional)

The silent fall-through in `dataset_metadata.py:367` should probably raise a clear error when `tasks.loc[task_string]` fails to find the index. As written, the failure leaks all the way to the policy-specific processor and presents as a confusing tokenizer error. A 2-line validator in `load_tasks()` that asserts `df.index.name == "task"` would prevent anyone else from hitting this.

## Local mitigation applied (for posterity)

For our running pi0/pi0.5 jobs against the `_target` dataset, the local cache at

```
/lustre-storage/datasets/zengh/genesis-hr-bench/lerobot_qpos_target_root/lerobot_datasets/genesis_hr_bench_lerobot_qpos_target/meta/tasks.parquet
```

was patched in place using the one-liner above. The original (broken) file is backed up at `tasks.parquet.bak_columns_layout` in the same dir. Once the HF source is fixed, this local override should be removed (delete the patched file + re-download from HF).
