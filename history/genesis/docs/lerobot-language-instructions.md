# LeRobot Language Instructions

This note compares the language instructions in the LeRobot dataset with the task variants used by `envs/`.
The LeRobot dataset stores task labels as instruction strings in `meta/tasks.parquet`; the environment maps those strings back to task classes through each class's `INSTRUCTION` constant.

## Variant Pattern

| Variant | Meaning | Typical language cue |
|---|---|---|
| Base / no-human | Robot-only version of the task. | Direct command with no human reference. |
| Assist | Human actively helps complete part of the task. | `while the human helps`, `human puts`, `human adds`, `work in parallel`. |
| Interrupt | Human interrupts or inspects an object mid-task. | `human inspects`, `wait`, `then continue`, `midway`. |
| Neutral | Human is present but doing nearby work. | `human works nearby on the same table`. |

## Families

### `dump_bin`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `dump_bin` | place each object into the trash can beside the table |
| Assist | `dump_bin_assist` | empty the table cubes into the trash bin while the human tosses another cube into the bin |
| Interrupt | `dump_bin_interrupt` | place each cube into the trash can; the human will inspect one cube midway through - wait, then continue. |
| Neutral | `dump_bin_neutral` | put the table objects into the bin while the human works nearby on the same table |

### `place_bread_in_basket`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `place_bread_in_basket` | put both pieces of bread into the basket |
| Assist | `place_bread_in_basket_assist` | put one bread into the basket while the human puts the other bread into the basket |
| Interrupt | `place_bread_in_basket_interrupt` | put both breads into the basket; if the human inspects one bread midway through, wait and then continue |
| Neutral | `place_bread_in_basket_neutral` | place both pieces of bread in the basket while the human works nearby on the same table |

### `place_burger_fries`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `place_burger_fries` | place the hamburger and side foods on the tray |
| Assist | `place_burger_fries_assist` | place the hamburger and side foods on the tray while the human adds one food item |
| Interrupt | `place_burger_fries_interrupt` | place the hamburger and side foods on the tray; if the human inspects one item midway through, wait and then continue |
| Neutral | `place_burger_fries_neutral` | place the meal items on the tray while the human works nearby on the same table |

### `place_dual_shoes`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `place_dual_shoes` | put the shoes in the shoebox |
| Assist | `place_dual_shoes_assist` | the human will put one shoe in the shoebox; the robot puts the other shoe in the shoebox - work in parallel |
| Interrupt | `place_dual_shoes_interrupt` | put the shoes in the shoebox; the human will inspect one shoe midway through - wait, then continue |
| Neutral | `place_dual_shoes_neutral` | place both shoes in the shoebox while the human works nearby on the same table |

### `place_food_in_skillet`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `place_food_in_skillet` | place the skillet on the stove and put the foods in it |
| Assist | `place_food_in_skillet_assist` | place the skillet on the stove and put the foods in it while the human helps place one food |
| Interrupt | `place_food_in_skillet_interrupt` | place the skillet on the stove and put the foods in it; the human may inspect one food item while you are reaching for it |
| Neutral | `place_food_in_skillet_neutral` | place the food in the skillet while the human works nearby on the same table |

### `put_object_cabinet`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `put_object_cabinet` | place the object inside the open cabinet drawer |
| Assist | `put_object_cabinet_assist` | place the object inside the human-opened cabinet drawer |
| Interrupt | `put_object_cabinet_interrupt` | place the object inside the cabinet drawer after the human inspects it |
| Neutral | `put_object_cabinet_neutral` | put the table object in the cabinet drawer while the human works nearby on the same table |

### `stack_bowls_three`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `stack_bowls_three` | stack the three bowls on top of each other |
| Assist | `stack_bowls_three_assist` | stack one bowl on the base bowl while the human stacks another bowl on the base |
| Interrupt | `stack_bowls_three_interrupt` | stack the three bowls on top of each other; if the human inspects one bowl midway through, wait and then continue |
| Neutral | `stack_bowls_three_neutral` | stack the bowls while the human works nearby on the same table |

### `blocks_ranking_rgb`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `blocks_ranking_rgb` | place the red block, green block, and blue block in a row from left to right |
| Assist | `blocks_ranking_rgb_assist` | place the red, green, and blue blocks in left-to-right order while the human helps place one block |
| Interrupt | `blocks_ranking_rgb_interrupt` | place the red, green, and blue blocks in left-to-right order; wait while the human inspects one block, then continue |
| Neutral | `blocks_ranking_rgb_neutral` | place the red, green, and blue blocks in left-to-right order while the human works nearby on the same table |

### `blocks_ranking_size`

| Variant | Env task | Language instruction |
|---|---|---|
| Base / no-human | `blocks_ranking_size` | place the largest, medium, and smallest blocks in a row from left to right |
| Assist | `blocks_ranking_size_assist` | place the RGB blocks in largest-to-smallest order while the human helps place one block |
| Interrupt | `blocks_ranking_size_interrupt` | place the RGB blocks in largest-to-smallest order; wait while the human inspects one block, then continue |
| Neutral | `blocks_ranking_size_neutral` | place the blocks from largest to smallest while the human works nearby on the same table |

### `categorize`

| Variant | Env task | Language instruction |
|---|---|---|
| Cooperative | `categorize_cooperative` | sort the objects into the matching baskets |
| Interrupt | `categorize_interrupt` | sort the objects into the matching baskets |
| Neutral | `categorize_neutral` | sort the objects into matching baskets while the human works nearby on the same table |

## Notes

- `categorize_cooperative` and `categorize_interrupt` share an instruction string. `categorize_interrupt` is kept as a compatibility alias outside `TASK_MAP` so the registry count matches the 44 unique LeRobot instruction rows.
- The no-human eval for `dump_bin_neutral` should use `--no-human`, which resolves the env task to `dump_bin` and therefore uses the base instruction: `place each object into the trash can beside the table`.
- The LeRobot target-abs dataset has 44 instruction strings, and `envs/tasks/__init__.py` now exposes 44 `TASK_MAP` entries by registering the base/no-human task classes from `envs/task_bases/`.
