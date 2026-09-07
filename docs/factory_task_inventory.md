# Genesis HR Bench — Task Inventory (non-base)

**42 non-base tasks** (the 10 no-human *base* tasks are excluded).

## A. Intention / singleton tasks (incl. factory work-cell)

| # | Task | Notes |
|---|------|-------|
| 1 | `categorize_cooperative` | Human + robot co-sort objects |
| 2 | `deliver_to_human_easy` | Deliver an object to a human's hand |
| 3 | `factory_inspect_pack` | Factory HRC — human inspects, robot packs (intent/cueing) |
| 4 | `factory_kit_packing` | Factory HRC — human+robot pack one kit in parallel |
| 5 | `factory_line_feeding` | Factory HRC — robot feeds a tool to the kneeling worker |
| 6 | `oil_bottle_recovery` | Recover a tipped oil bottle |
| 7 | `open_microwave` | Open the microwave door |
| 8 | `pour_water` | Robot pours water |
| 9 | `stamp_documents` | Robot stamps documents |
| 10 | `take_from_human_easy` | Take an object from a human |
| 11 | `take_from_human_safety` | Take-from-human with safety |

## B. Variant families (assist / interrupt / neutral)

| # | Family | assist | interrupt | neutral |
|---|--------|:------:|:---------:|:-------:|
| 1 | `blocks_ranking_rgb` | ✅ | ✅ | ✅ |
| 2 | `blocks_ranking_size` | ✅ | ✅ | ✅ |
| 3 | `categorize` | — | — | ✅ |
| 4 | `dump_bin` | ✅ | ✅ | ✅ |
| 5 | `dump_bin_xarm_calibrated` | ✅ | ✅ | — |
| 6 | `place_bread_in_basket` | ✅ | ✅ | ✅ |
| 7 | `place_burger_fries` | ✅ | ✅ | ✅ |
| 8 | `place_dual_shoes` | ✅ | ✅ | ✅ |
| 9 | `place_food_in_skillet` | ✅ | ✅ | ✅ |
| 10 | `put_object_cabinet` | ✅ | ✅ | ✅ |
| 11 | `stack_bowls_three` | ✅ | ✅ | ✅ |
| 12 | `suitcase_organize` | — | ✅ | — |

**Family-variant totals:** assist 10 · interrupt 11 · neutral 10 = 31; plus 11 intention/singleton = **42**.

