"""Verify bounded episodes finalize native outcomes even without simulation steps."""

import importlib.util
import os
from pathlib import Path
from unittest import TestCase, mock

path = Path(__file__).resolve().parents[1] / "evaluation/policies/vlm_agent/deploy.py"
spec = importlib.util.spec_from_file_location("vlm_deploy_limits", path)
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class FakeEnv:
    task_name = "test"
    step_lim = 100

    def __init__(self):
        self.success = [True]
        self.take_action_cnt = [0]
        self.deploy_cfg = {"vlm_agent": {"max_decisions": 3, "max_zero_progress": 2}}

    def is_episode_end(self):
        return not self.success[0]

    def get_obs(self):
        return {}

    def take_action(self, action):
        self.take_action_cnt[0] += 1


class FakeClient:
    def __init__(self, actions):
        self.actions = actions
        self.reports = []
        self.calls = 0

    def call(self, func_name, obs=None):
        if func_name == "get_action":
            self.calls += 1
            return self.actions
        if func_name == "report_execution":
            self.reports.append(obs)


class EpisodeLimitsTest(TestCase):
    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": "{}"})
    def test_empty_policy_result_is_not_false_positive_success(self):
        env, client = FakeEnv(), FakeClient([])
        deploy.eval_one_episode(env, client)
        self.assertFalse(env.success[0])
        self.assertEqual(client.reports[-1]["termination_reason"], "policy_stop")

    @mock.patch.dict(os.environ, {"VLM_AGENT_OVERRIDES": '{"episode_budget_s": 0}'})
    def test_wall_budget_does_not_start_another_api_call(self):
        env, client = FakeEnv(), FakeClient([{}])
        deploy.eval_one_episode(env, client)
        self.assertEqual(client.calls, 0)
        self.assertFalse(env.success[0])
        self.assertEqual(client.reports[-1]["termination_reason"], "wall_budget")
