import unittest

from artificium.usage import cache_hit_ratio, normalize_usage


class NormalizeUsageCase(unittest.TestCase):
    def test_openai_chat_with_cache_and_reasoning(self):
        value = normalize_usage({
            "prompt_tokens": 137012, "completion_tokens": 4593, "total_tokens": 141605,
            "prompt_tokens_details": {"cached_tokens": 4096},
            "completion_tokens_details": {"reasoning_tokens": 3741},
        })
        self.assertEqual(value["input_tokens"], 137012)
        self.assertEqual(value["output_tokens"], 4593)
        self.assertEqual(value["reasoning_tokens"], 3741)
        self.assertEqual(value["cache_read_tokens"], 4096)
        self.assertIsNone(value["cache_write_tokens"])
        self.assertAlmostEqual(cache_hit_ratio(value), 4096 / 137012)

    def test_openai_responses(self):
        value = normalize_usage({
            "input_tokens": 1000, "output_tokens": 50,
            "input_tokens_details": {"cached_tokens": 900},
            "output_tokens_details": {"reasoning_tokens": 20},
        })
        self.assertEqual((value["input_tokens"], value["cache_read_tokens"], value["reasoning_tokens"]), (1000, 900, 20))

    def test_anthropic_adds_cache_buckets_to_input(self):
        value = normalize_usage({
            "input_tokens": 10, "output_tokens": 30,
            "cache_read_input_tokens": 800, "cache_creation_input_tokens": 190,
        })
        self.assertEqual(value["input_tokens"], 1000)
        self.assertEqual(value["cache_read_tokens"], 800)
        self.assertEqual(value["cache_write_tokens"], 190)
        self.assertEqual(value["total_tokens"], 1030)

    def test_gemini_adds_thoughts_to_output(self):
        value = normalize_usage({
            "promptTokenCount": 500, "candidatesTokenCount": 40, "thoughtsTokenCount": 60,
            "cachedContentTokenCount": 300, "totalTokenCount": 600,
        })
        self.assertEqual((value["input_tokens"], value["output_tokens"], value["reasoning_tokens"],
                          value["cache_read_tokens"], value["total_tokens"]), (500, 100, 60, 300, 600))

    def test_ollama(self):
        value = normalize_usage({"prompt_eval_count": 12, "eval_count": 7})
        self.assertEqual((value["input_tokens"], value["output_tokens"]), (12, 7))
        self.assertIsNone(cache_hit_ratio(value))

    def test_llamacpp_timings_supply_cache(self):
        value = normalize_usage(
            {"prompt_tokens": 1200, "completion_tokens": 10},
            {"timings": {"cache_n": 1100, "prompt_n": 100, "predicted_n": 10}},
        )
        self.assertEqual(value["cache_read_tokens"], 1100)
        self.assertEqual(value["input_tokens"], 1200)

    def test_separate_reasoning_total_is_added(self):
        value = normalize_usage({"completion_tokens": 10, "reasoning_tokens": 5})
        self.assertEqual((value["output_tokens"], value["reasoning_tokens"]), (15, 5))

    def test_missing_usage(self):
        value = normalize_usage(None)
        self.assertTrue(all(item is None for item in value.values()))


if __name__ == "__main__":
    unittest.main()


class VersionCase(unittest.TestCase):
    def test_single_version_source(self):
        import artificium
        from artificium.engine import DEFAULT_USER_AGENT
        from artificium.runtime import Artificium
        from artificium.version import RELEASE, VERSION

        self.assertEqual(artificium.VERSION, VERSION)
        self.assertEqual(Artificium.VERSION, VERSION)
        self.assertIn(RELEASE, DEFAULT_USER_AGENT)
