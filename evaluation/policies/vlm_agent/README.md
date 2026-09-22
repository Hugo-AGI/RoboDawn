# RoboDojo VLM policy

The RoboDawn controller as a RoboDojo policy: an external VLM drives RoboDojo tasks through discrete commands, scored by
RoboDojo's native reward. The policy is an XPolicyLab policy: a policy server
(this directory, in the `vlm_policy` environment) that talks to the model, and
a simulation-side loop (`deploy.py`, imported by the Isaac Sim client in the
`RoboDojo` environment) that plans and executes the commands. All commands
below run from the repository root.

## Installation

```bash
conda create -n vlm_policy python=3.11 -y && conda activate vlm_policy
pip install "numpy<2" pyyaml opencv-python-headless h5py "websockets>=14" \
    msgpack msgpack-numpy pydantic pillow -r evaluation/policies/vlm_agent/requirements.txt
bash scripts/robodojo/setup_vlm_policy.sh
```

`setup_vlm_policy.sh` links this directory into `RoboDojo/XPolicyLab/policy/`,
checks the prerequisites and re-renders the robot configs of the asset bundle
(`scripts/robodojo/fix_asset_paths.sh`). The simulator uses RoboDojo's own
conda environment; the policy server has its own because it needs
`websockets>=14`.

Credentials: copy `secrets.example.json` to `secrets.json` at the repository
root, one profile per model with `name`, `model`, `base_url` and `api_key`.
A profile is selected by `--model` (alias `--profile`), `VLM_AGENT_PROFILE`,
`deploy.yml`'s `vlm_profile`, or the first entry.

## Running

```bash
bash scripts/robodojo/run_vlm_eval.sh --list-models
bash scripts/robodojo/run_vlm_eval.sh --model gpt-6-astra --task stack_bowls --env-gpu 0
python scripts/robodojo/run_vlm_experiment.py run --experiment-dir experiments/robodojo \
    --run-id gpt6_stack_bowls --model gpt-6-astra --task stack_bowls --gpu 0
python scripts/robodojo/run_vlm_experiment.py summary --experiment-dir experiments/robodojo
```

`run_vlm_eval.sh` is the plain launch through RoboDojo's `robodojo.sh eval`;
`run_vlm_experiment.py` wraps it with a run directory (manifest, launcher
log, per-turn decision logs, `summary.json`) and a
`summary` command that folds every run of an experiment into `results.csv`.
Native results and videos land in `RoboDojo/eval_result/RoboDojo/<task>/vlm_agent/`.
INT/TERM/HUP stop the run and clean up the process group it created.

Every knob comes from [deploy.yml](deploy.yml); `VLM_AGENT_OVERRIDES` (a JSON
object) overrides top-level keys for one run, and `run_vlm_experiment.py` uses
it to pass its arguments. The shipped `deploy.yml` holds the configuration the
reported results were produced with: 240 turns, 8000 reply tokens,
`reasoning_effort: high`, a 660 s decision budget, a 9000 s episode budget,
the head camera and both wrist cameras, the overlays on, gripper measurement
on, and the demonstration bank `demos/robodojo`.

Request errors (network, timeout, 400/408/429/5xx) are retried up to
`max_attempts` (6) with exponential backoff, honouring `Retry-After`; 401/403/404
end the episode. A turn whose request fails after its retries is a turn error;
`max_consecutive_errors` (2) of them end the episode. `decision_budget_s`
covers the request and its retries, and the simulation RPC timeout is raised
above it.

## Commands

The model answers each turn with a JSON object `{scene, progress, memory,
plan, commands}`; `commands` is a list of strings in this grammar
([commands.py](commands.py)):

```
<arm> move x|y|z <cm>            translate the fingertip centre along a world axis, keeping the orientation, |cm| <= 20
<arm> rotate roll|pitch|yaw <deg> rotate about a world axis through the fingertip centre, 15 deg steps, |deg| <= 90
<arm> point down|down15|down30|down45|down60|down75|forward
                                 orientation presets: fingers straight down, tilted towards +y, or straight forward
<arm> gripper open|close|0..1    gripper
<arm> home                       return to the initial pose
wait / done
```

Non-zero rotations are clipped and then rounded to the nearest 15 degree
step; the execution feedback records both. The first `max_commands_per_turn`
(4) commands of a turn are executed in order by the simulation client
([main_route.py](main_route.py)); every command is resolved from the arm's live
pose at that moment, so a failed command leaves the next one relative to where
the arm really is.

* The control point is the fingertip centre, 14.5 cm ahead of the wrist
  RoboDojo reports; `rotate` turns about it.
* Every motion is planned by RoboDojo's cuRobo planner (`curobo_motion.py`),
  which shares the IK solver and the coordinate conventions of RoboDojo's own
  tasks. Its collision world is the table, not the objects on it. A planning
  failure is reported before the arm moves, together with whether IK found a
  joint solution at all (out of reach) or only no collision-free path.
* A command counts as reached within 1.5 cm / 8 deg of its target; otherwise
  the result says how far the fingertips actually moved. The final joint
  target is held for `motion_settle_steps` (10) control steps before the
  check.
* `home` first tries the planner; if the planner refuses, the arm returns
  along a direct joint-space path to the joints recorded on the first turn,
  and the result says so.
* `done` never ends the episode by itself. On the first `done` of a run both
  arms are sent home (RoboDojo's checkers expect the arms back at their
  origin; the policy removes the "then reset the robot arms" sentence from
  the instruction instead). After `done_limit` (3) consecutive `done`
  turns without success the episode ends.
* Memory: the last `history_turns` (12) turns as machine-written history plus
  the model's own scratchpad, both in every prompt.

The prompt is built from [profiles/robodojo_x5_main.yaml](profiles/robodojo_x5_main.yaml)
(the robot, the frame in centimetres, the reachable region, the cameras, the
gripper and a few tips, with `{table_z}` placeholders) and the grammar; see
[main_prompts.py](main_prompts.py).

## Budget

An episode is bounded by `max_decisions` (240) model turns and by the time
budgets (`decision_budget_s`, `episode_budget_s`). RoboDojo's own control-step
limit is sized for policies that act every 25 Hz frame; one command here is a
whole planned motion of 10-30 control steps, so `deploy.py` lifts that limit
and the step counter in the state is informational.

## Gripper measurement

`measure_gripper: true` reads the gripper joint before every turn and puts it
in the observation as `gripper_real`. RoboDojo's `<side>_ee_joint_state` is
the last applied command, not the finger position, so it reads the same
whether or not an object stopped the fingers; the joint itself is what
RoboDojo's own reward checks read (`is_all_gripper_open`). After a close, an
opening more than `grasp_open_threshold` (0.05) above the empty-closed
reading is reported to the model as "something is between the fingers"; the
height at which that happened is kept in the prompt so the model can compute a
release height. The gripper is only read once it has settled
([gripper_settle.py](gripper_settle.py)).

## Visual aids

Drawn on the head image ([grounding.py](grounding.py)), each switchable in
`visual_aids`: `markers` (wrist circle, fingertip-centre cross and the line
between them, per arm), `grid` (10 cm grid on the table plane with 5 cm minor
lines, X/Y labels and an axis legend) and `enhance` (percentile contrast
stretch). The wrist images get the fingertip cross, 5 cm world-axis arrows
with 1 cm ticks and the jaw axis. The overlays need the camera calibration the
simulation client attaches to the observation; if it is missing the aid is
dropped with one warning.

## Demonstrations

The demonstration of the task being run (`<demo_bank>/<task>/demo.json`, see
`demos/README.md`) is rendered into one user message placed between the
system prompt and every turn: the selected phases with their images, the
fingertip state, the phase description and the commands. The system prompt
tells the model to copy the strategy, not the numbers. The frames the model
saw are saved next to the decision logs (`icl_demo/`). A bank that does not
exist, or has no entry for the task, fails the launch.

## Rendering compatibility

On hosts with NVIDIA 595 drivers run `python3 scripts/robodojo/compat/vulkan_driver_compat.py --prepare`
once; the launcher then wraps the simulation client with the Khronos
validation layer that corrects the driver's allocation limit. Verified
dependencies go to `.cache/robodojo-vulkan`; the system driver is not
modified. `ROBODOJO_VULKAN_COMPAT=off` disables the check.
