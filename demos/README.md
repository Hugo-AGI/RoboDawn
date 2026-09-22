# Demonstrations

Everything the model sees in context, in the rendered form the policies read.
Nothing here is generated at evaluation time: the banks are inputs of the
reported runs and are shipped so that the runs can be repeated exactly.

## `robotwin2/` (RoboTwin 2.0)

* `primer/`: the task-independent command primer, six steps that show what
  each command type does in a clean scene from the start pose (image before
  the commands, state, explanation, commands). Every RoboTwin episode receives
  it first.
* `expert/<task>/` and `expert/<task>+2/`: one or two demonstrations per task
  (85 entries for the 50 tasks), each a successful episode of the task in a
  `demo_clean` scene. `demo.json` lists the selected key turns
  (first turn, every turn with a gripper or orientation command, the last
  turn, the final image) with the overview image, the state line, the plan,
  the commands and their object-relative net effect; `grasps` records on
  which table half each arm grasped; at the first turn of an episode the agent
  picks the entry whose grasp sides match the scene.

Demonstrations come from `demo_clean` scenes and evaluation uses
`demo_randomized` (the seeds in `harness/valid_seeds/`); the source seed of
every demonstration is recorded in its `demo.json`.

## `robodojo/` (RoboDojo)

One entry per task, `robodojo/<task>/demo.json` plus the JPEG frames it
names. Each entry is one successful episode of the task on a layout that is
not one of the five evaluation layouts, packed as its turns (40 for most
tasks, 6 to 49 across the bank): per frame the phase label (the task rule and
the running state), the fingertip state, the plan, the commands and their
effect, plus a head-camera image for at most 16 of the frames.

`MANIFEST.json` records, per task, the MD5 of the shipped `demo.json` and the
number measured with it: successes over the five evaluation layouts of seed 0
(layouts 0-4; for `imitate_sorting_sequence` RoboDojo classified layout 4 as
unstable on every attempt and, as its runner does, layout 5 was measured
instead) and RoboDojo's score.

| Task | Dimension | Success | Score | Frames |
| --- | --- | --- | --- | --- |
| align_blocks | Open | 4/5 | 80 | 14 |
| classify_objects_by_language | Open | 1/5 | 44 | 40 |
| general_pickup | Open | 5/5 | 100 | 6 |
| pour_by_language | Open | 5/5 | 100 | 40 |
| solve_equation | Open | 5/5 | 100 | 10 |
| stack_blocks_by_language | Open | 5/5 | 100 | 14 |
| store_tools_in_toolbox | Open | 0/5 | 20 | 40 |
| pick_from_conveyor_by_image | Open | 0/5 | 0 | 15 |
| cover_blocks | Memory | 5/5 | 100 | 40 |
| imitate_sorting_sequence | Memory | 3/5 | 66 | 40 |
| match_and_pick_from_conveyor | Memory | 2/5 | 40 | 10 |
| press_by_number | Memory | 5/5 | 100 | 40 |
| swap_T | Memory | 1/5 | 20 | 40 |
| swap_blocks | Memory | 0/5 | 0 | 26 |
| classify_objects | Long-Horizon | 4/5 | 83 | 40 |
| make_kong | Long-Horizon | 3/5 | 60 | 40 |
| organize_table | Long-Horizon | 1/5 | 75 | 40 |
| play_stacking_toy | Long-Horizon | 0/5 | 10 | 40 |
| play_tic_tac_toe | Long-Horizon | 5/5 | 100 | 40 |
| put_bottles_into_dustbin | Long-Horizon | 4/5 | 88 | 40 |
| fill_egg_holder | Long-Horizon | 1/5 | 37 | 45 |
| fill_pen_holder | Long-Horizon | 0/5 | 26 | 48 |
| arrange_largest_number | Generalization | 5/5 | 100 | 40 |
| fold_clothes | Generalization | 1/5 | 36 | 40 |
| hang_mugs | Generalization | 0/5 | 14 | 40 |
| make_toast | Generalization | 1/5 | 20 | 40 |
| pack_objects_into_box | Generalization | 0/5 | 19 | 40 |
| pour_liquid_into_cup | Generalization | 4/5 | 80 | 27 |
| push_T | Generalization | 2/5 | 40 | 40 |
| sort_nesting_dolls_by_size | Generalization | 4/5 | 80 | 40 |
| stack_blocks | Generalization | 5/5 | 100 | 32 |
| stack_bowls | Generalization | 5/5 | 100 | 21 |
| sweep_blocks | Generalization | 3/5 | 60 | 40 |
| store_laptop_and_headphones | Generalization | 0/5 | 12 | 49 |
| build_tower | Precision | 2/5 | 54 | 14 |
| deposit_coin | Precision | 0/5 | 16 | 10 |
| fasten_screws | Precision | 0/5 | 34 | 40 |
| insert_tubes | Precision | 4/5 | 88 | 48 |
| play_Xylophone | Precision | 4/5 | 80 | 40 |
| plug_in_charger | Precision | 0/5 | 0 | 35 |
| pour_balls_into_vase | Precision | 0/5 | 0 | 38 |
| insert_key | Precision | 0/5 | 15 | 31 |
