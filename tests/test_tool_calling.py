"""Tests for the OpenAI tool-calling compatibility layer."""

import json
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


openai_format = _load_module("openai_format", os.path.join(ROOT, "server", "openai_format.py"))
tool_calling = _load_module("tool_calling", os.path.join(ROOT, "server", "tool_calling.py"))

tool_calls_response = openai_format.tool_calls_response
build_tool_prompt = tool_calling.build_tool_prompt
parse_tool_calls = tool_calling.parse_tool_calls


class ToolCallingTests(unittest.TestCase):
    def test_parse_plain_tool_calls_json(self):
        calls = parse_tool_calls(
            '{"tool_calls":[{"name":"get_weather","arguments":{"city":"Singapore"}}]}'
        )

        self.assertEqual(calls[0]["type"], "function")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Singapore"})

    def test_parse_fenced_tool_calls_json(self):
        calls = parse_tool_calls(
            '```json\n{"tool_calls":[{"function":{"name":"lookup","arguments":"{\\"q\\":\\"x\\"}"}}]}\n```'
        )

        self.assertEqual(calls[0]["function"]["name"], "lookup")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"q": "x"})

    def test_normal_text_is_not_a_tool_call(self):
        self.assertIsNone(parse_tool_calls("The answer is 42."))

    def test_build_prompt_records_tool_choice(self):
        prompt = build_tool_prompt(
            "User: hi",
            [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            {"type": "function", "function": {"name": "lookup"}},
            False,
        )

        self.assertIn("Tool choice requires calling only this tool: lookup.", prompt)
        self.assertIn("Call at most one tool.", prompt)

    def test_tool_calls_response_shape(self):
        resp = tool_calls_response(
            [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
            "copilot",
            "conv_1",
        )

        choice = resp["choices"][0]
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"][0]["id"], "call_test")


if __name__ == "__main__":
    unittest.main()
