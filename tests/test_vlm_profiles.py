"""VLM profile selection and request routing without credentials or networking."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


CLIENT_PATH = Path(__file__).resolve().parents[1] / "evaluation/policies/vlm_agent/vlm_client.py"
SPEC = importlib.util.spec_from_file_location("vlm_client", CLIENT_PATH)
vlm_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vlm_client)


def profile(name="fast", model="model-fast", **fields):
    return {
        "name": name,
        "model": model,
        "base_url": "https://private-endpoint.invalid/v1",
        "api_key": "private-test-api-key",
        **fields,
    }


class VLMProfilesTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.secrets = Path(self.tempdir.name) / "secrets.json"
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def write_config(self, data):
        self.secrets.write_text(json.dumps(data), encoding="utf-8")
        return str(self.secrets)

    def load(self, name=None):
        return vlm_client.load_vlm_profile(name, str(self.secrets))

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLIENT_PATH), "--secrets", str(self.secrets), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_list_order_alias_model_id_and_environment_priority(self):
        self.write_config({"vlm": [profile(), profile("accurate", "model-accurate")]})
        profiles = vlm_client.load_vlm_profiles(str(self.secrets))
        self.assertEqual([item["name"] for item in profiles], ["fast", "accurate"])
        self.assertEqual(self.load()["name"], "fast")
        self.assertEqual(self.load("accurate")["model"], "model-accurate")
        self.assertEqual(self.load("model-accurate")["name"], "accurate")
        os.environ[vlm_client.PROFILE_ENV_VAR] = "accurate"
        self.assertEqual(self.load()["name"], "accurate")
        self.assertEqual(self.load("fast")["name"], "fast")

    def test_profile_name_takes_priority_over_model_id(self):
        self.write_config([profile("first", "chosen"), profile("chosen", "second-model")])
        self.assertEqual(self.load("chosen")["model"], "second-model")

    def test_shared_model_id_requires_a_profile_name(self):
        self.write_config([profile("gateway-a", "same-model"), profile("gateway-b", "same-model")])
        with self.assertRaisesRegex(vlm_client.VLMError, "ambiguous"):
            self.load("same-model")
        self.assertEqual(self.load("gateway-b")["name"], "gateway-b")

    def test_duplicate_names_are_rejected_before_selection(self):
        self.write_config([profile("same", "first"), profile("same", "second")])
        with self.assertRaisesRegex(vlm_client.VLMError, "duplicate"):
            self.load("first")

    def test_alternative_shapes_and_unnamed_profiles(self):
        unnamed = profile()
        del unnamed["name"]
        cases = [
            ({"vlm": unnamed}, "model-fast"),
            (unnamed, "model-fast"),
            ({"vlm": {"single": unnamed}}, "single"),
            ({"single": unnamed}, "single"),
            ([unnamed], "model-fast"),
        ]
        for config, expected_name in cases:
            with self.subTest(config=config):
                self.write_config(config)
                self.assertEqual(self.load()["name"], expected_name)

    def test_all_profiles_are_validated(self):
        for invalid in [
            profile(api_key=""),
            profile(model=123),
            profile(name=None),
            profile(name="line\nbreak"),
            profile(extra_body=["invalid"]),
            "invalid-entry",
        ]:
            with self.subTest(invalid_type=type(invalid).__name__):
                self.write_config({"vlm": [profile(), invalid]})
                with self.assertRaises(vlm_client.VLMError):
                    self.load("fast")

    def test_unknown_selection_is_reported(self):
        self.write_config([profile()])
        with self.assertRaisesRegex(vlm_client.VLMError, "not found"):
            self.load("missing")
        os.environ[vlm_client.PROFILE_ENV_VAR] = "missing"
        with self.assertRaisesRegex(vlm_client.VLMError, "not found"):
            self.load()

    def test_explicit_path_and_environment_do_not_fall_back(self):
        self.write_config([profile()])
        missing = str(self.secrets.parent / "missing.json")
        os.environ[vlm_client.SECRETS_ENV_VAR] = str(self.secrets)
        with self.assertRaises(vlm_client.VLMError):
            vlm_client.load_vlm_profiles(missing)
        self.assertEqual(vlm_client._candidate_secret_paths(missing), [Path(missing)])
        os.environ[vlm_client.SECRETS_ENV_VAR] = missing
        with self.assertRaises(vlm_client.VLMError):
            vlm_client.load_vlm_profiles()
        self.assertEqual(vlm_client._candidate_secret_paths(), [Path(missing)])
        self.assertEqual(self.load()["name"], "fast")

    def test_empty_and_invalid_json_raise_configuration_errors(self):
        for contents in ["{malformed-private-test-api-key", "{\"vlm\": []}", "null"]:
            with self.subTest(contents=contents):
                self.secrets.write_text(contents, encoding="utf-8")
                with self.assertRaises(vlm_client.VLMError) as caught:
                    self.load()
                self.assertNotIn("private-test-api-key", str(caught.exception))

    def test_cli_listing_and_selection_do_not_print_secrets(self):
        self.write_config([profile(), profile("accurate", "model-accurate")])
        listing = self.cli("--list-models")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn("fast", listing.stdout)
        self.assertIn("model-accurate", listing.stdout)
        self.assertEqual(listing.stderr, "")
        for option in ("--model", "--profile"):
            selection = self.cli(option, "model-accurate")
            self.assertEqual(selection.returncode, 0, selection.stderr)
            self.assertEqual(selection.stdout, "accurate\n")
        failure = self.cli("--model", "missing")
        self.assertNotEqual(failure.returncode, 0)
        self.assertEqual(failure.stdout, "")
        self.assertIn("not found", failure.stderr)
        for result in (listing, selection, failure):
            output = result.stdout + result.stderr
            self.assertNotIn("private-test-api-key", output)
            self.assertNotIn("private-endpoint", output)
            self.assertNotIn("Traceback", output)

    def test_invalid_config_cli_is_concise_and_secret_safe(self):
        self.secrets.write_text('{"vlm": private-test-api-key}', encoding="utf-8")
        result = self.cli("--list-models")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("invalid JSON", result.stderr)
        self.assertNotIn("private-test-api-key", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_selected_model_cannot_be_overridden_by_extra_body(self):
        self.write_config([
            profile(),
            profile("accurate", "model-accurate", extra_body={"model": "wrong", "top_p": 0.9}),
        ])
        client = vlm_client.VLMClient(self.load("accurate"))
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode())
        with patch.object(vlm_client.urllib.request, "urlopen", return_value=response) as urlopen:
            result = client.complete([{"role": "user", "content": "hello"}])
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], "model-accurate")
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(result["text"], "ok")


if __name__ == "__main__":
    unittest.main()
