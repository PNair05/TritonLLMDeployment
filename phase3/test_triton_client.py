import unittest
from unittest.mock import patch

from triton_client import TritonClient, format_mistral_instruction


class FakeResponse:
    def __init__(self, body: str):
        self.body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self.body


class TritonClientTests(unittest.TestCase):
    def test_formats_mistral_instruction(self):
        self.assertEqual(format_mistral_instruction(" hello "), "[INST] hello [/INST]")

    def test_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            format_mistral_instruction("   ")

    @patch("urllib.request.urlopen")
    def test_generate_uses_generate_extension(self, urlopen):
        urlopen.return_value = FakeResponse('{"text_output": "  hello  "}')
        client = TritonClient("http://example.test")

        self.assertEqual(client.generate("Hi", 8), "hello")
        request = urlopen.call_args.args[0]
        payload = request.data.decode("utf-8")
        self.assertEqual(request.full_url, "http://example.test/v2/models/ensemble/generate")
        self.assertIn('"text_input": "[INST] Hi [/INST]"', payload)
        self.assertIn('"max_tokens": 8', payload)


if __name__ == "__main__":
    unittest.main()
