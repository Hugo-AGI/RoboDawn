"""Malformed replies and unsafe coercions of the reply schema."""

import importlib.util
import json
import pathlib
import unittest

HAS_JSON5 = importlib.util.find_spec("json5") is not None


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "vlm_action_parser", ROOT / "evaluation/policies/vlm_agent/action_parser.py",
)
PARSER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER)


class ActionParserTests(unittest.TestCase):
    def parse(self, raw):
        return PARSER.parse_action_response(raw if isinstance(raw, str) else json.dumps(raw))

    def assert_rejected(self, raw):
        parsed, metadata = self.parse(raw)
        self.assertIsNone(parsed)
        self.assertFalse(metadata["recovered"])
        self.assertTrue(metadata["error"])

    def test_strict_json_and_text_fields(self):
        response = {"scene": "two bowls", "progress": "starting", "memory": "", "plan": "approach",
                    "commands": ["left point down", "left move z -5"]}
        parsed, metadata = self.parse(response)
        self.assertEqual(parsed, response)
        self.assertEqual(metadata, {"parser": "json", "recovered": False, "repairs": [], "repair_details": [], "error": None})
        self.assertEqual(self.parse({"commands": ["wait"]})[0], {"commands": ["wait"]})

    def test_accepted_command_field_shapes(self):
        self.assertEqual(self.parse({"commands": "left point down; wait"})[0]["commands"], ["left point down", "wait"])
        self.assertEqual(self.parse({"commands": "left point down\nwait\n"})[0]["commands"], ["left point down", "wait"])
        self.assertEqual(self.parse({"commands": [{"command": "wait"}, {"cmd": "done"}]})[0]["commands"], ["wait", "done"])
        self.assertEqual(self.parse({"commands": {"command": "wait"}})[0]["commands"], ["wait"])
        parsed, _ = self.parse({"commands": ["wait"], "scene": {"objects": 2}, "plan": ["a", "b"]})
        self.assertEqual(parsed["scene"], '{"objects": 2}')
        self.assertEqual(parsed["plan"], '["a", "b"]')

    def test_commands_are_required_and_bounded(self):
        for commands in (None, [], "", [""], [1], [{"note": "x"}], ["wait"] * 17, {"note": "x"}):
            with self.subTest(commands=commands):
                self.assert_rejected({"scene": "s", "commands": commands})
        self.assert_rejected({"scene": "s"})
        self.assertIsNotNone(self.parse({"commands": ["wait"] * 16})[0])

    def test_generic_extractor_tolerates_fences_and_prose(self):
        raw = 'sure!\n```json\n{"commands": ["wait"]}\n```\ndone'
        self.assertEqual(PARSER.extract_json_object(raw), {"commands": ["wait"]})
        self.assertEqual(PARSER.extract_json_object('{"memory": {"phase": "align"}}'), {"memory": {"phase": "align"}})
        self.assertEqual(self.parse(raw)[0], {"commands": ["wait"]})

    @unittest.skipUnless(HAS_JSON5, "numeric recovery needs the json5 dependency")
    def test_leading_plus_numbers_are_repaired_and_recorded_with_offsets(self):
        raw = '```json\n{"scene":"dx: +0.20; quote \\\" and { braces }", "turn":+2, "commands":["left move x +2"]}\n```'
        parsed, metadata = self.parse(raw)
        self.assertEqual(parsed["turn"], 2)
        self.assertEqual(parsed["commands"], ["left move x +2"])
        self.assertIn("+0.20", parsed["scene"])
        self.assertEqual(metadata["parser"], "json5_numeric")
        self.assertTrue(metadata["recovered"])
        self.assertEqual(len(metadata["repairs"]), 1)
        self.assertEqual(metadata["repair_details"], [{
            "kind": "numeric_leading_plus", "offset": raw.index("+2,"),
            "original": "+2", "normalized": "2",
        }])

    @unittest.skipUnless(HAS_JSON5, "numeric recovery needs the json5 dependency")
    def test_positive_decimal_and_exponent_numbers(self):
        for token, expected in (("+0.02", 0.02), ("+0", 0), ("+12", 12), ("+2e-2", 0.02), ("+2E+2", 200)):
            with self.subTest(token=token):
                parsed, metadata = self.parse('{"value":' + token + ', "commands":["wait"]}')
                self.assertEqual(parsed["value"], expected)
                self.assertTrue(metadata["recovered"])

    def test_damaged_outer_object_never_returns_nested_content(self):
        for raw in (
            '{"commands":["wait"],"value":???,"memory":"align"}',
            '{"commands":["wait"],"memory":"align"',
            '{"broken":, "nested":{"commands":["wait"]}}',
            '{"commands":["wait"],}',
            '{"value":+0.02,,"commands":["wait"]}',
            '[{"commands":["wait"]}]',
            '{"wrapper":{"commands":["wait"]}}',
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw)

    def test_several_top_level_objects_yield_the_first_answer_object(self):
        # a reply may hold several top-level objects; the one with the answer keys is taken
        parsed, _ = PARSER.parse_action_response('{"notes":{"phase":"align"}} {"commands":["left point down"]}')
        self.assertEqual(parsed["commands"], ["left point down"])
        parsed, _ = PARSER.parse_action_response('{"commands":["wait"]} {"commands":["done"]}')
        self.assertEqual(parsed["commands"], ["wait"])

    def test_recovery_does_not_accept_non_numeric_json5_syntax(self):
        for raw in (
            "{'commands': ['wait'], 'value': +0.02}",
            '{commands:["wait"],"value":+0.02}',
            '{"commands":["wait"],"value":+0.02 /* note */}',
            '{"commands":["wait"],"value":+0.02, // note\n"done":false}',
            '{"scene":"bad\\x41 escape","commands":["wait"],"value":+0.02}',
            '{"commands":["wait"],"value":+}',
            '{"commands":["wait"],"value":01}',
            '{"commands":["wait"],"value":+01}',
            '{"commands":["wait"],"value":+0x10}',
            '{"commands":["wait"],"value":0x10}',
            '{"commands":["wait"],"value":+.02}',
            '{"commands":["wait"],"value":.02}',
            '{"commands":["wait"],"value":+1.}',
            '{"commands":["wait"],"value":1.}',
            '{"commands":["wait"],"value":+1_000}',
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw)

    def test_duplicate_keys_rejected_at_all_depths(self):
        for raw in (
            '{"commands":["wait"],"commands":["done"]}',
            '{"commands":["wait"],"memory":{"phase":"align","phase":"lift"}}',
        ):
            with self.subTest(raw=raw):
                self.assert_rejected(raw)

    def test_nonfinite_numbers_are_rejected_anywhere(self):
        for token in ("NaN", "Infinity", "-Infinity", "+Infinity", "1e999", "+1e999"):
            with self.subTest(token=token):
                self.assert_rejected('{"commands":["wait"],"value":' + token + '}')
        self.assert_rejected('{"commands":["wait"],"memory":{"value":1e999}}')

    def test_empty_non_object_and_oversized_inputs(self):
        for raw in (None, "", "no action", "true", '"{\\"commands\\": [\\"wait\\"]}"', "{" * 200_001):
            with self.subTest(raw=repr(raw)[:50]):
                self.assert_rejected(raw)


if __name__ == "__main__":
    unittest.main()
