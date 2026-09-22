# harness: the RoboDawn controller on RoboTwin 2.0

Closed-loop control of a robot by a multimodal LLM through discrete commands.

The robot (simulated or real) is wrapped in a small discrete command interface.
Every turn the model sees the images and the state, replies with a few
commands, the commands are executed, and the model sees the result. The
controller is robot-agnostic: RoboTwin 2.0 is one environment implementation,
and a real robot needs one environment subclass plus one profile.

```
harness/
├── core/                 robot-agnostic core
│   ├── commands.py       command syntax and parser (move / rotate / point / gripper / home / wait / done)
│   ├── env.py            DiscreteEnvBase: the environment interface (observe / execute / state / views / should_stop)
│   └── watchdog.py       stall watchdog: dumps the stacks and exits when a process makes no progress
├── agent/                the LLM controller
│   ├── llm_client.py     OpenAI-compatible client: retries, logging, per-family reasoning fields, 400 / rate-limit fallbacks
│   ├── prompts.py        system prompt = profile + grammar + reply format; reply JSON parsing
│   ├── memory.py         history (written by the harness) + scratchpad (written by the model)
│   ├── demos.py          in-context demonstrations: bank loading, side selection, rendering into messages
│   └── mllm_agent.py     the closed-loop episode, grasp facts, end conditions
├── robotwin/env.py       RoboTwin 2.0 behind DiscreteEnvBase (official protocol, cameras, overlays, video)
├── examples/real_robot_skeleton.py   real-robot skeleton (with --mock dry run)
├── configs/              robotwin2_profile.yaml (the adaptation artefact), real_robot_profile_template.yaml, task list
├── valid_seeds/          expert-validated evaluation seeds (official RoboTwin protocol), shipped with the results
├── run_robotwin_eval.py  RoboTwin evaluation entry point
└── scripts/aggregate_results.py  per-task and overall success rate of a run directory
```

## Command grammar (what the model emits)

```
<arm> move x|y|z <cm>           translate the fingertips along a world axis, keeping the orientation, |cm| <= 20
<arm> rotate roll|pitch|yaw <deg> rotate about a world axis through the fingertips, |deg| <= 90
<arm> point down|forward|down45 orientation presets
<arm> gripper open|close        gripper
<arm> home                      return to the initial pose
wait / done
```

## Design points

These were established in simulation and hold on a real robot just the same.

1. **The control point is the fingertip centre (TCP), not the wrist.**
   RoboTwin's `endpose` is the wrist; the fingertips are 12 cm further along
   the approach direction, and getting this wrong hits the table.
2. **One command is one complete planned motion**, with the magnitude chosen
   by the model; execution runs to rest before the next images are taken, so an
   episode needs only 10-30 decisions.
3. **Honest feedback**: a planning failure tells the model "the arm did not
   move"; the gripper is read only after it has actually settled, because a
   half-closed gripper looks like a grasp.
4. **Make coordinates readable**: an overview camera plus a 10 cm grid on the
   table plane and fingertip markers bring the model's position estimates from
   5-10 cm error to 2-3 cm.
5. **`done` does not end the episode**: if the checker has not registered
   success the model is told the task is not finished.
6. **Memory**: a scratchpad the model rewrites every turn helps; a longer
   machine-written history alone does not.

## Running the RoboTwin 2.0 evaluation

Prerequisites, once:

1. The `RoboTwin/` submodule at the pinned commit, installed per its README
   (conda environment `RoboTwin` with SAPIEN and cuRobo) with its assets
   downloaded. Nothing in the submodule is modified. Another official checkout
   of the same commit can be used through `ROBOTWIN_ROOT`.
2. An API key file, e.g. `~/.config/robodawn/key` (one key per line, several
   keys are spread over processes), or `LLM_API_KEY`; the endpoint through
   `--api_base` or `LLM_API_BASE`.
3. The evaluation assets shipped in this repository: `harness/valid_seeds/`
   (the seeds of every episode, the same ones the reported results used; do
   not delete or rebuild them), `demos/robotwin2/primer/` and
   `demos/robotwin2/expert/` (see `demos/README.md`).

One task (the defaults are the reported configuration: `demo_randomized`
scenes, 45 turns, one expert demonstration chosen by grasp side plus the
primer, reasoning on, 8000 reply tokens):

```bash
conda activate RoboTwin
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python harness/run_robotwin_eval.py \
    --task place_empty_cup --episodes 10 --model gpt-6-astra \
    --api_base https://<endpoint>/v1 --api_key_file ~/.config/robodawn/key \
    --output results/rt2/gpt-6-astra/place_empty_cup/shard_0
python harness/scripts/aggregate_results.py results/rt2          # per task + overall success rate
```

An episode takes 10-30 minutes, so run tasks in parallel: three processes fit
on one 24 GB GPU (5.7 GB each). Runs are resumable: finished episodes in
`--output` are skipped. The 50 task names are in
`harness/configs/robotwin2_all_tasks.txt`.

Notes:

* **Reasoning must really be on.** The client adds per-family fields
  (`llm_client.py::THINKING_EXTRA`); after a first episode check that
  `usage.completion_tokens_details.reasoning_tokens` in `llm_calls.jsonl` is in
  the hundreds to thousands. The same model behaves very differently behind a
  gateway that silently drops reasoning. `--reasoning_effort` overrides the
  family default when an endpoint only accepts particular values.
* **Images per request**: 6 primer + 4 current turn + every demonstration
  frame (at most 48), i.e. up to 58 images; some gateways cap requests at 50.
* Compare only on the same seeds: `harness/valid_seeds/` is shipped and is
  only ever appended to. Every episode counts: one whose `finished_reason` is
  `model_error` or `parse_failure` (the endpoint did not answer) is a failure,
  in the reported numbers and in `aggregate_results.py` alike.

The demonstrations are placed as one user message (followed by a short
assistant acknowledgement) between the system prompt and the current turn.
Each shows the key turns of a successful episode: the overview image, the
state, the plan, the commands and their object-relative net effect, with the
failed commands removed and a note on why that arm was chosen. The
demonstration each episode received is saved in `<run>/<model>/<task>/shard_k/demo*/`,
and `results.json` records its source seed. Profiles write table heights as
`{table_z}` / `{table_z+5}` placeholders, filled per episode from the
environment's `prompt_context()`.

## Putting the controller on a real robot

1. Copy `examples/real_robot_skeleton.py` and implement its five hardware
   methods: read the fingertip pose (rotation matrix column 0 = approach,
   column 1 = finger axis), read the real gripper opening, take the images,
   plan and execute to a target pose (return False without moving on failure),
   set the gripper and wait for it to stop.
2. Dry-run with `--mock` first to check the API, the prompt and the parsing,
   then connect the hardware.
3. Fill in `configs/real_robot_profile_template.yaml`: how to read the frame,
   the reachable region (measure it on a grid with the real planner, do not
   guess), the camera descriptions, grasp-height constants.
4. Draw the fingertip markers and the table grid in `views()` (needs camera
   calibration); this is what makes positions readable.
5. Success: connect the `success` property to your checker, a keyboard
   confirmation or a VLM judge; `should_stop()` to a safety stop.
6. Safety: command magnitudes are capped in `core/commands.py::LIMITS`; add
   workspace clipping and speed limits in your `_move_to`.
