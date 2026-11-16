"""Prices, the per-day ledger, the daily limit and what the operator sees."""

from __future__ import annotations

import contextlib
import dataclasses
import datetime as dt
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

try:
    from test_revolution import FakeEngine, _seed_install
except ImportError:  # run as tests.test_cost
    from tests.test_revolution import FakeEngine, _seed_install

from artificium import cost
from artificium.cli import main
from artificium.config import Config, ConfigStore, MODEL_FIELDS, OVERSIGHT_FIELDS, SecretsStore
from artificium.cost import CostLedger, CostLimitReached
from artificium.engine import EngineError, EngineReply
from artificium.filesystem import Paths, read_json
from artificium.operator import format_status, status_snapshot
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.usage import normalize_usage

PRICES = dict(price_input=3.0, price_cached_input=0.3, price_cache_write=3.75,
              price_output=15.0)
JUDGE = {"provider": "anthropic", "model": "judge-model", "context_window_tokens": 30_000,
         "price_input": 1.0, "price_output": 5.0}


def _config(**changes) -> Config:
    base = dict(provider="custom", model="worker-model", base_url="http://worker.invalid/v1",
                context_window_tokens=20_000, context_reminder_tokens=2_000)
    return Config(**{**base, **changes})


class FakeClock:
    """A settable epoch, so no test waits for a day to pass."""

    def __init__(self, local: dt.datetime):
        self.now = local.timestamp()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class PriceTests(unittest.TestCase):
    def test_plain_input_and_output_are_priced_per_million(self) -> None:
        usage = normalize_usage({"prompt_tokens": 2_000_000, "completion_tokens": 100_000})
        self.assertAlmostEqual(cost.compute_cost(_config(**PRICES), usage), 6.0 + 1.5)

    def test_cached_and_cache_write_tokens_use_their_own_prices(self) -> None:
        usage = normalize_usage({"input_tokens": 100_000, "cache_read_input_tokens": 800_000,
                                 "cache_creation_input_tokens": 100_000,
                                 "output_tokens": 10_000})
        self.assertEqual(usage["input_tokens"], 1_000_000)
        expected = 0.1 * 3.0 + 0.8 * 0.3 + 0.1 * 3.75 + 0.01 * 15.0
        self.assertAlmostEqual(cost.compute_cost(_config(**PRICES), usage), expected)

    def test_openai_style_cached_tokens_are_part_of_the_prompt(self) -> None:
        usage = normalize_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 400_000}})
        self.assertAlmostEqual(cost.compute_cost(_config(**PRICES), usage), 0.6 * 3.0 + 0.4 * 0.3)

    def test_a_missing_cache_price_bills_those_tokens_as_input(self) -> None:
        usage = normalize_usage({"prompt_tokens": 1_000_000, "completion_tokens": 0,
                                 "prompt_tokens_details": {"cached_tokens": 400_000}})
        config = _config(price_input=3.0, price_output=15.0)
        self.assertAlmostEqual(cost.compute_cost(config, usage), 3.0)

    def test_reasoning_tokens_are_billed_as_output(self) -> None:
        usage = normalize_usage({"prompt_tokens": 0, "completion_tokens": 1_000_000,
                                 "completion_tokens_details": {"reasoning_tokens": 600_000}})
        self.assertEqual(usage["reasoning_tokens"], 600_000)
        self.assertAlmostEqual(cost.compute_cost(_config(**PRICES), usage), 15.0)
        # Gemini reports thoughts beside the candidates; they still bill as output.
        usage = normalize_usage({"promptTokenCount": 0, "candidatesTokenCount": 400_000,
                                 "thoughtsTokenCount": 600_000})
        self.assertAlmostEqual(cost.compute_cost(_config(**PRICES), usage), 15.0)

    def test_without_prices_or_usage_the_cost_is_unknown(self) -> None:
        usage = normalize_usage({"prompt_tokens": 10, "completion_tokens": 5})
        self.assertIsNone(cost.compute_cost(_config(), usage))
        self.assertIsNone(cost.compute_cost(_config(price_input=1.0), usage))
        self.assertIsNone(cost.compute_cost(_config(**PRICES), normalize_usage({})))
        self.assertEqual(cost.price_usage(_config(), {"prompt_tokens": 10}, usage), (None, "unknown"))

    def test_a_cost_reported_by_the_provider_wins_over_the_computed_one(self) -> None:
        raw = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "cost": 0.42}
        config = _config(provider="openrouter", model="m", **PRICES)
        self.assertEqual(config.adapter, "openrouter")
        self.assertEqual(cost.price_usage(config, raw, normalize_usage(raw)), (0.42, "reported"))
        # Without a reported figure the configured prices are used.
        raw.pop("cost")
        self.assertEqual(cost.price_usage(config, raw, normalize_usage(raw)), (3.0, "computed"))
        # A reporting provider needs no prices at all.
        raw["cost"] = 0.5
        self.assertEqual(cost.price_usage(_config(provider="openrouter", model="m"), raw,
                                          normalize_usage(raw)), (0.5, "reported"))

    def test_a_cost_field_is_ignored_for_providers_that_do_not_report_one(self) -> None:
        raw = {"prompt_tokens": 1_000_000, "completion_tokens": 0, "cost": 99}
        self.assertEqual(cost.price_usage(_config(**PRICES), raw, normalize_usage(raw)),
                         (3.0, "computed"))


class LedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        self.paths.ensure_layout()

    def ledger(self, clock, **changes) -> CostLedger:
        return CostLedger(self.paths, _config(**changes), clock)

    def test_the_day_rolls_over_at_local_midnight_in_the_configured_zone(self) -> None:
        zone = ZoneInfo("Pacific/Auckland")
        clock = FakeClock(dt.datetime(2026, 3, 10, 23, 30, tzinfo=zone))
        ledger = self.ledger(clock, cost_day_timezone="Pacific/Auckland", daily_cost_limit=10.0)
        ledger.record("worker", {"input_tokens": 10, "output_tokens": 5}, 4.0)
        status = ledger.status()
        self.assertEqual((status["day"], status["spent"]), ("2026-03-10", 4.0))
        # Reset is the next local midnight, and that is not midnight UTC.
        self.assertEqual(status["reset_epoch"], dt.datetime(2026, 3, 11, tzinfo=zone).timestamp())
        self.assertEqual(status["reset_at"], "2026-03-11 00:00")
        clock.advance(31 * 60)
        status = ledger.status()
        self.assertEqual((status["day"], status["spent"]), ("2026-03-11", 0.0))
        ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 1.0)
        report = ledger.report(7)
        self.assertEqual([(d["day"], d["spent"]) for d in report["days"]],
                         [("2026-03-11", 1.0), ("2026-03-10", 4.0)])

    def test_the_same_instant_is_a_different_day_in_another_zone(self) -> None:
        # 2026-06-01 02:00 in Tokyo is still 2026-05-31 in Los Angeles' evening.
        instant = dt.datetime(2026, 6, 1, 2, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        tokyo = self.ledger(FakeClock(instant), cost_day_timezone="Asia/Tokyo")
        los_angeles = self.ledger(FakeClock(instant), cost_day_timezone="America/Los_Angeles")
        self.assertEqual(tokyo.status()["day"], "2026-06-01")
        self.assertEqual(los_angeles.status()["day"], "2026-05-31")

    def test_the_machine_zone_is_the_default(self) -> None:
        with mock.patch.dict(os.environ, {"TZ": "Pacific/Auckland"}):
            import time
            time.tzset()
            self.addCleanup(time.tzset)
            clock = FakeClock(dt.datetime(2026, 3, 10, 23, 30, tzinfo=ZoneInfo("Pacific/Auckland")))
            status = self.ledger(clock).status()
            self.assertEqual((status["day"], status["reset_at"], status["timezone"]),
                             ("2026-03-10", "2026-03-11 00:00", "local time"))

    def test_tokens_requests_cost_and_unknown_cost_are_kept_per_connection(self) -> None:
        clock = FakeClock(dt.datetime(2026, 1, 5, 12, 0, tzinfo=ZoneInfo("UTC")))
        ledger = self.ledger(clock, cost_day_timezone="UTC")
        usage = {"input_tokens": 100, "output_tokens": 20, "reasoning_tokens": 8,
                 "cache_read_tokens": 40, "cache_write_tokens": 10}
        ledger.record("worker", usage, 0.5)
        ledger.record("worker", usage, None)
        ledger.record("oversight", usage, 0.25, currency="EUR")
        entry = ledger.status()["connections"]
        self.assertEqual(entry["worker"]["requests"], 2)
        self.assertEqual(entry["worker"]["input_tokens"], 200)
        self.assertEqual(entry["worker"]["reasoning_tokens"], 16)
        self.assertEqual(entry["worker"]["cache_read_tokens"], 80)
        self.assertEqual(entry["worker"]["unknown_cost_requests"], 1)
        self.assertEqual((entry["oversight"]["cost"], entry["oversight"]["currency"]), (0.25, "EUR"))
        self.assertEqual(ledger.status()["spent"], 0.75)

    def test_the_ledger_keeps_a_bounded_number_of_days(self) -> None:
        clock = FakeClock(dt.datetime(2026, 1, 1, 12, 0, tzinfo=ZoneInfo("UTC")))
        ledger = self.ledger(clock, cost_day_timezone="UTC")
        for _ in range(cost.KEEP_DAYS + 20):
            ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 0.01)
            clock.advance(86_400)
        self.assertEqual(len(read_json(ledger.path)["days"]), cost.KEEP_DAYS)
        self.assertLess(ledger.path.stat().st_size, 40_000)

    def test_a_damaged_ledger_file_starts_again_rather_than_failing(self) -> None:
        self.paths.runtime.mkdir(parents=True, exist_ok=True)
        (self.paths.runtime / "cost-ledger.json").write_text("{not json")
        ledger = self.ledger(FakeClock(dt.datetime(2026, 1, 1, tzinfo=ZoneInfo("UTC"))),
                             cost_day_timezone="UTC")
        self.assertEqual(ledger.status()["spent"], 0.0)

    def test_the_header_line_shows_spend_only_when_there_is_something_to_show(self) -> None:
        clock = FakeClock(dt.datetime(2026, 1, 5, 12, 0, tzinfo=ZoneInfo("UTC")))
        ledger = self.ledger(clock, cost_day_timezone="UTC")
        self.assertEqual(cost.header_line(ledger.config, ledger.status(), False), "")
        self.assertEqual(cost.header_line(ledger.config, ledger.status(), True),
                         "Spend today: 0.00 USD")
        ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 0.5)
        self.assertEqual(cost.header_line(ledger.config, ledger.status(), False),
                         "Spend today: 0.50 USD")
        limited = self.ledger(clock, cost_day_timezone="UTC", daily_cost_limit=2.0)
        self.assertEqual(cost.header_line(limited.config, limited.status(), False),
                         "Spend today: 0.50 of 2.00 USD limit (25%)")
        ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, None)
        self.assertIn("1 requests of unknown cost not counted",
                      cost.header_line(ledger.config, ledger.status(), False))


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))

    def test_prices_belong_to_the_model_section_and_the_limit_to_the_harness(self) -> None:
        for name in (*cost.PRICE_FIELDS, "price_currency"):
            self.assertIn(name, MODEL_FIELDS)
            self.assertIn(name, OVERSIGHT_FIELDS)
        config = _config(**PRICES, price_currency="EUR", daily_cost_limit=12.5,
                         cost_day_timezone="Europe/Paris",
                         oversight_connection=dict(JUDGE, price_currency="EUR"))
        ConfigStore(self.paths).save(config)
        saved = read_json(self.paths.config)
        self.assertEqual(saved["model"]["price_input"], 3.0)
        self.assertEqual(saved["harness"]["daily_cost_limit"], 12.5)
        self.assertEqual(saved["harness"]["cost_day_timezone"], "Europe/Paris")
        self.assertEqual(saved["oversight"]["price_output"], 5.0)
        self.assertEqual(ConfigStore(self.paths).load(), config)
        self.assertEqual(config.oversight_config().price_input, 1.0)
        # The oversight connection has its own prices, never the worker's.
        self.assertIsNone(_config(**PRICES, oversight_connection={
            "provider": "anthropic", "model": "j"}).oversight_config().price_input)

    def test_every_field_is_written_and_a_missing_key_means_the_default(self) -> None:
        ConfigStore(self.paths).save(_config())
        saved = read_json(self.paths.config)
        for name in cost.PRICE_FIELDS:
            self.assertIsNone(saved["model"][name])
        self.assertEqual(saved["model"]["price_currency"], "USD")
        self.assertIsNone(saved["harness"]["daily_cost_limit"])
        self.assertIsNone(saved["harness"]["cost_day_timezone"])
        # A file written before the settings existed loads with no prices, no limit.
        for section, names in (("model", (*cost.PRICE_FIELDS, "price_currency")),
                               ("harness", ("daily_cost_limit", "cost_day_timezone"))):
            for name in names:
                del saved[section][name]
        loaded = Config.from_dict(saved)
        self.assertEqual((loaded.price_input, loaded.daily_cost_limit, loaded.price_currency),
                         (None, None, "USD"))

    def test_a_setting_in_the_wrong_section_is_a_load_error(self) -> None:
        grouped = _config().grouped_dict()
        for place, name, value in (("harness", "price_input", 1.0),
                                   ("harness", "price_currency", "USD"),
                                   ("model", "daily_cost_limit", 5.0),
                                   ("model", "cost_day_timezone", "UTC")):
            broken = json.loads(json.dumps(grouped))
            broken[place][name] = value
            with self.assertRaises(ValueError, msg=f"{place}.{name}"):
                Config.from_dict(broken)

    def test_invalid_values_are_refused(self) -> None:
        for changes in ({"price_input": -1}, {"price_output": "free"}, {"price_input": True},
                        {"price_currency": ""}, {"daily_cost_limit": 0},
                        {"daily_cost_limit": -3}, {"cost_day_timezone": "Mars/Olympus"}):
            with self.assertRaises(ValueError, msg=str(changes)):
                _config(**changes)
        with self.assertRaises(ValueError):
            _config(oversight_connection=dict(JUDGE, price_input=-2))

    def test_a_limit_needs_prices_or_a_provider_that_reports_costs(self) -> None:
        problem = cost.cost_limit_problem(_config(daily_cost_limit=5.0))
        self.assertIn("worker", problem)
        self.assertIn("configure model --price-input AMOUNT --price-output AMOUNT", problem)
        self.assertIsNone(cost.cost_limit_problem(_config(**PRICES, daily_cost_limit=5.0)))
        self.assertIsNone(cost.cost_limit_problem(_config(price_input=0, price_output=0,
                                                          daily_cost_limit=5.0)))
        self.assertIsNone(cost.cost_limit_problem(
            _config(provider="openrouter", model="m", daily_cost_limit=5.0)))
        self.assertIsNone(cost.cost_limit_problem(_config()))
        # Both connections are covered.
        unpriced_judge = _config(**PRICES, daily_cost_limit=5.0, oversight_connection={
            "provider": "anthropic", "model": "judge"})
        self.assertIn("configure oversight --price-input", cost.cost_limit_problem(unpriced_judge))
        by_name = _config(**PRICES, daily_cost_limit=5.0, oversight_model="other-model")
        self.assertIn("other-model", cost.cost_limit_problem(by_name))
        mixed = _config(**PRICES, daily_cost_limit=5.0,
                        oversight_connection=dict(JUDGE, price_currency="EUR"))
        self.assertIn("currency", cost.cost_limit_problem(mixed))


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        _seed_install(self.paths)
        ConfigStore(self.paths).save(_config())
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--root", str(self.paths.install), *argv])
        return code, out.getvalue(), err.getvalue()

    def config(self) -> Config:
        return ConfigStore(self.paths).load()

    def test_a_limit_without_prices_is_refused_and_says_what_to_set(self) -> None:
        code, _, err = self.run_cli("configure", "harness", "--daily-cost-limit", "5")
        self.assertEqual(code, 2)
        self.assertIn("--price-input", err)
        self.assertIn("--price-output", err)
        self.assertIsNone(self.config().daily_cost_limit)

    def test_prices_are_set_offline_per_connection_then_the_limit_is_accepted(self) -> None:
        code, out, err = self.run_cli(
            "configure", "model", "--price-input", "3", "--price-cached-input", "0.3",
            "--price-cache-write", "3.75", "--price-output", "15", "--price-currency", "EUR")
        self.assertEqual(code, 0, err)
        worker = self.config()
        self.assertEqual((worker.price_input, worker.price_cached_input, worker.price_cache_write,
                          worker.price_output, worker.price_currency), (3.0, 0.3, 3.75, 15.0, "EUR"))
        self.assertIn("Prices, worker: input 3, cached input 0.3", out)
        # Nothing else was touched and no model was asked.
        self.assertEqual((worker.provider, worker.model), ("custom", "worker-model"))
        code, out, err = self.run_cli("configure", "harness", "--daily-cost-limit", "20",
                                      "--cost-day-timezone", "Europe/Paris")
        self.assertEqual(code, 0, err)
        saved = self.config()
        self.assertEqual((saved.daily_cost_limit, saved.cost_day_timezone), (20.0, "Europe/Paris"))
        self.assertEqual(saved.price_input, 3.0)
        self.assertIn("Daily cost limit: 20 EUR", out)
        code, _, _ = self.run_cli("configure", "harness", "--daily-cost-limit", "off",
                                  "--cost-day-timezone", "local")
        self.assertEqual(code, 0)
        self.assertEqual((self.config().daily_cost_limit, self.config().cost_day_timezone), (None, None))

    def test_prices_without_a_scope_apply_to_the_worker_offline(self) -> None:
        code, _, err = self.run_cli("configure", "--price-input", "2", "--price-output", "8")
        self.assertEqual(code, 0, err)
        self.assertEqual((self.config().price_input, self.config().price_output), (2.0, 8.0))

    def test_the_interactive_menu_sets_prices_and_the_limit(self) -> None:
        from artificium.setup import SetupOptions, SetupWizard
        ConfigStore(self.paths).save(_config(oversight_connection=dict(JUDGE)))
        answers = iter([
            # worker: currency, input, cached input, cache write, output
            "EUR", "3", "0.3", "unset", "15",
            # oversight: currency, input, cached input, cache write, output
            "EUR", "1", "unset", "unset", "5",
            "20", "Europe/Paris", "y",
        ])
        with mock.patch("builtins.input", side_effect=lambda *_: next(answers)), \
                contextlib.redirect_stdout(io.StringIO()):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="cost"), interactive=True)
        saved = self.config()
        self.assertEqual((saved.price_input, saved.price_cached_input, saved.price_cache_write,
                          saved.price_output, saved.price_currency), (3.0, 0.3, None, 15.0, "EUR"))
        self.assertEqual((saved.daily_cost_limit, saved.cost_day_timezone), (20.0, "Europe/Paris"))
        self.assertEqual(saved.oversight_config().price_output, 5.0)

    def test_oversight_prices_are_set_on_the_oversight_connection_only(self) -> None:
        code, _, err = self.run_cli("configure", "oversight", "--price-input", "1")
        self.assertEqual(code, 2)
        self.assertIn("No separate oversight connection", err)
        ConfigStore(self.paths).save(_config(**PRICES, oversight_connection=dict(JUDGE)))
        code, out, err = self.run_cli("configure", "oversight", "--price-input", "2",
                                      "--price-output", "off")
        self.assertEqual(code, 0, err)
        saved = self.config()
        self.assertEqual((saved.price_input, saved.price_output), (3.0, 15.0))
        separate = saved.oversight_config()
        self.assertEqual((separate.price_input, separate.price_output), (2.0, None))
        self.assertIn("Prices, oversight: input 2", out)
        # With the limit set, removing a covered connection's price is refused.
        self.assertEqual(self.run_cli("configure", "oversight", "--price-output", "5")[0], 0)
        self.assertEqual(self.run_cli("configure", "harness", "--daily-cost-limit", "9")[0], 0)
        code, _, err = self.run_cli("configure", "oversight", "--price-input", "off")
        self.assertEqual(code, 2)
        self.assertIn("oversight", err)
        self.assertEqual(self.config().oversight_config().price_input, 2.0)

    def test_flags_are_checked_against_their_scope(self) -> None:
        self.assertEqual(self.run_cli("configure", "harness", "--price-input", "1")[0], 2)
        self.assertEqual(self.run_cli("configure", "model", "--daily-cost-limit", "1")[0], 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_cli("configure", "model", "--price-input", "-1")
        self.assertEqual(self.run_cli("configure", "harness", "--cost-day-timezone", "Nowhere/Land")[0], 2)

    def test_the_cost_command_and_status_show_spend_limit_and_reset(self) -> None:
        ConfigStore(self.paths).save(_config(**PRICES, daily_cost_limit=10.0,
                                             cost_day_timezone="UTC"))
        code, out, _ = self.run_cli("cost")
        self.assertEqual(code, 0)
        self.assertIn("No model requests recorded yet.", out)
        self.assertIn("Daily limit: 10.00 USD", out)
        ledger = CostLedger(self.paths, self.config())
        ledger.record("worker", {"input_tokens": 1000, "output_tokens": 50,
                                 "cache_read_tokens": 500}, 2.5)
        ledger.record("oversight", {"input_tokens": 10, "output_tokens": 5}, None)
        code, out, _ = self.run_cli("cost", "--days", "3")
        self.assertIn("spent 2.50 (25%)", out)
        self.assertIn("worker: 1 requests; 1,000 in (500 cached read", out)
        self.assertIn("oversight: 1 requests", out)
        self.assertIn("1 requests of unknown cost", out)
        self.assertIn("Resets at", out)
        self.assertIn("estimates", out)
        data = json.loads(self.run_cli("cost", "--json")[1])
        self.assertEqual(data["today"]["spent"], 2.5)
        code, out, _ = self.run_cli("status")
        self.assertIn("Spend today: 2.50 of 10.00 USD (25%)", out)

    def test_status_says_when_requests_are_held_by_the_limit(self) -> None:
        state = {"process": {"alive": True, "owned": True, "pid": 1}, "configured": True,
                 "root": "x", "provider": "p", "model": "m", "working_context_tokens": 1,
                 "context_window_tokens": 1, "working_memory_tokens": 1, "auto_repair": True,
                 "unhandled_interactions": [], "scheduled_tasks": [], "vision_preference": "auto",
                 "effective_vision": "auto", "mandatory_offload": False, "sleep": {},
                 "offload_pending": False, "offload_threshold_percent": 80, "cost": None,
                 "runtime": {"status": "cost_limited", "reset_at": "2026-03-11 00:00",
                             "timezone": "UTC"}}
        text = format_status(state)
        self.assertIn("held by the daily cost limit", text)
        self.assertIn("Held until 2026-03-11 00:00 (UTC)", text)


class RuntimeCase(unittest.TestCase):
    """An instance whose ledger clock is a fake, on a worker and a judge engine."""

    config_changes: dict = {}
    judge = True

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        _seed_install(self.paths)
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        changes = dict(PRICES, daily_cost_limit=10.0, cost_day_timezone="Asia/Tokyo",
                       **self.config_changes)
        if self.judge:
            changes["oversight_connection"] = dict(JUDGE)
        ConfigStore(self.paths).save(_config(**changes))
        SecretsStore(self.paths).save_api_key("worker-key")
        self.worker = FakeEngine([])
        self.judge_engine = FakeEngine([])
        patch = mock.patch("artificium.runtime.make_engine", return_value=self.judge_engine)
        patch.start()
        self.addCleanup(patch.stop)
        self.agent = Artificium(self.paths, engine=self.worker, console=Console(quiet=True))
        self.clock = FakeClock(dt.datetime(2026, 5, 4, 20, 0, tzinfo=ZoneInfo("Asia/Tokyo")))
        self.agent.cost.clock = self.clock
        self.agent.initialization.finish("The test instance is initialized and ready for its task.")

    def spend(self, amount: float, connection: str = "worker") -> None:
        self.agent.cost.record(connection, {"input_tokens": 1, "output_tokens": 1}, amount)

    def lifetime(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]

    def ask(self, text: str = "work") -> None:
        self.agent._complete([{"role": "user", "content": text}], wake_reason="test")


class ResponseAccountingTests(RuntimeCase):
    def test_a_response_is_priced_recorded_and_added_to_the_ledger(self) -> None:
        self.worker.complete = lambda messages: EngineReply(
            "<think>ok</think>", usage={"prompt_tokens": 1_000_000, "completion_tokens": 100_000})
        self.ask()
        record = self.lifetime("model_response")[-1]
        self.assertAlmostEqual(record["cost"], 4.5)
        self.assertEqual((record["cost_source"], record["cost_currency"]), ("computed", "USD"))
        status = self.agent.cost.status()
        self.assertAlmostEqual(status["spent"], 4.5)
        self.assertEqual(status["connections"]["worker"]["requests"], 1)

    def test_the_oversight_connection_is_priced_with_its_own_prices(self) -> None:
        self.judge_engine.complete = lambda messages: EngineReply(
            "VERDICT: ok", usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        self.agent._oversight_complete([{"role": "user", "content": "judge"}])
        status = self.agent.cost.status()
        self.assertAlmostEqual(status["connections"]["oversight"]["cost"], 6.0)
        self.assertNotIn("worker", status["connections"])
        self.assertEqual(self.lifetime("model_response")[-1]["connection"], "oversight")

    def test_a_provider_reported_cost_is_used_and_labelled(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, provider="openrouter",
                                                model="m", **{})
        self.worker.complete = lambda messages: EngineReply(
            "<think>ok</think>", usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.07})
        self.ask()
        record = self.lifetime("model_response")[-1]
        self.assertEqual((record["cost"], record["cost_source"]), (0.07, "reported"))
        self.assertEqual(self.agent.cost.status()["spent"], 0.07)

    def test_without_prices_the_tokens_count_and_the_cost_is_unknown(self) -> None:
        self.agent.config = dataclasses.replace(
            self.agent.config, price_input=None, price_output=None, daily_cost_limit=None)
        self.worker.complete = lambda messages: EngineReply(
            "<think>ok</think>", usage={"prompt_tokens": 10, "completion_tokens": 5})
        self.ask()
        record = self.lifetime("model_response")[-1]
        self.assertEqual((record["cost"], record["cost_source"]), (None, "unknown"))
        entry = self.agent.cost.status()["connections"]["worker"]
        self.assertEqual((entry["input_tokens"], entry["output_tokens"], entry["unknown_cost_requests"]),
                         (10, 5, 1))
        self.assertEqual(entry["cost"], 0.0)

    def test_a_failed_request_the_provider_billed_is_counted(self) -> None:
        def fail(messages):
            raise EngineError("cut off", kind="context", reply=EngineReply(
                "", usage={"prompt_tokens": 1_000_000, "completion_tokens": 0}))

        self.worker.complete = fail
        with self.assertRaises(EngineError):
            self.ask()
        record = self.lifetime("model_request_failed")[-1]
        self.assertAlmostEqual(record["cost"], 3.0)
        self.assertAlmostEqual(self.agent.cost.status()["spent"], 3.0)

        def fail_unbilled(messages):
            raise EngineError("refused", status=503)

        self.worker.complete = fail_unbilled
        with self.assertRaises(EngineError):
            self.ask()
        self.assertNotIn("cost", self.lifetime("model_request_failed")[-1])
        self.assertEqual(self.agent.cost.status()["connections"]["worker"]["requests"], 1)

    def test_the_other_oversight_model_on_the_workers_endpoint_does_not_inherit_prices(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, oversight_connection=None,
                                                oversight_model="other")
        target, _ = self.agent._oversight_target()
        self.assertEqual(target.model, "other")
        self.assertIsNone(target.price_input)


class StateHeaderTests(RuntimeCase):
    def header(self) -> str:
        return self.agent._state_header("test", 1_000)

    def test_the_header_shows_spend_against_the_limit(self) -> None:
        self.spend(2.5)
        self.assertIn("Spend today: 2.50 of 10.00 USD limit (25%)\n", self.header())

    def test_the_header_has_no_cost_line_when_there_is_nothing_to_show(self) -> None:
        self.agent.config = dataclasses.replace(
            self.agent.config, price_input=None, price_output=None, daily_cost_limit=None,
            oversight_connection=None)
        self.agent.cost.config = self.agent.config
        self.assertNotIn("Spend today", self.header())
        self.assertIn("Context status: normal\nPending interaction focus: none\n", self.header())


class WarningTests(RuntimeCase):
    def test_the_notice_comes_once_a_day_at_eighty_percent(self) -> None:
        self.spend(7.9)
        self.assertEqual(self.agent._cost_events(), [])
        self.spend(0.2)
        events = self.agent._cost_events()
        self.assertEqual(len(events), 1)
        text = events[0]
        self.assertIn("8.10 of 10.00 USD (81%)", text)
        self.assertIn("1.90 USD remain", text)
        self.assertIn("2026-05-05 00:00 (Asia/Tokyo)", text)
        self.assertEqual(self.agent._cost_events(), [])
        self.assertEqual(len(self.lifetime("cost_warning_issued")), 1)
        # The next day it can come again.
        self.clock.advance(5 * 3600)
        self.assertEqual(self.agent._cost_events(), [])
        self.spend(8.5)
        self.assertEqual(len(self.agent._cost_events()), 1)

    def test_the_notice_reaches_the_next_request_of_the_life_loop(self) -> None:
        self.spend(8.5)
        inputs, claimed, _ = self.agent._collect_round_inputs(
            turn_id="t", trigger="notification", continuous=True, round_number=1,
            state={"pulse_used": False, "first_wake_issued": True,
                   "meta_memory_guidance_issued": True, "guided_meta_memory": None})
        self.assertTrue(any("DAILY COST LIMIT" in item for item in inputs))

    def test_no_notice_without_a_limit(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, daily_cost_limit=None)
        self.agent.cost.config = self.agent.config
        self.spend(500)
        self.assertEqual(self.agent._cost_events(), [])


class LimitTests(RuntimeCase):
    # These tests count oversight requests; a decision's review would be one.
    config_changes = {"decision_review": False}

    def test_at_the_limit_no_request_is_sent_from_either_connection(self) -> None:
        self.spend(6.0)
        self.spend(4.0, "oversight")  # the limit covers both together
        with self.assertRaises(CostLimitReached) as caught:
            self.ask()
        self.assertIn("2026-05-05 00:00", str(caught.exception))
        with self.assertRaises(CostLimitReached):
            self.agent._oversight_complete([{"role": "user", "content": "judge"}])
        self.assertEqual((len(self.worker.requests), len(self.judge_engine.requests)), (0, 0))
        self.assertEqual(self.lifetime("model_request"), [])
        # A refusal is not a connection failure.
        self.assertEqual(self.lifetime("oversight_connection_failed"), [])
        # The limit is the sum: just under it, requests go out.
        self.clock.advance(4 * 3600 + 1)
        self.ask()
        self.assertEqual(len(self.worker.requests), 1)

    def test_a_request_in_flight_when_the_limit_is_crossed_is_not_cancelled(self) -> None:
        self.spend(9.0)
        self.worker.complete = lambda messages: EngineReply(
            "<think>ok</think>", usage={"prompt_tokens": 1_000_000, "completion_tokens": 100_000})
        self.ask()
        self.assertTrue(self.agent.cost.status()["limited"])
        self.assertEqual(self.lifetime("model_response")[-1]["content"], "<think>ok</think>")
        with self.assertRaises(CostLimitReached):
            self.ask()

    def test_review_and_consult_tools_report_the_limit_to_the_agent(self) -> None:
        self.spend(10.0)
        result = self.agent.tools.consult("Should the parser be rewritten or patched?")
        self.assertEqual(result["status"], "error")
        self.assertIn("daily cost limit", result["summary"])
        self.assertEqual(len(self.judge_engine.requests), 0)

    def test_oversight_checks_wait_and_are_not_counted_as_failures(self) -> None:
        from artificium.interactions import ArtificiumClient
        ArtificiumClient(self.paths.install).send("room", sender="entity_1", content="Please build it.")
        self.agent.tools.send_interaction("room", "Which one?", decision=True, default="proceed")
        self.spend(10.0)
        later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=30)
        self.assertEqual(self.agent.oversight.tick(later, force=True), [])
        oversight = read_json(self.paths.runtime / "oversight.json", {})
        self.assertEqual(oversight.get("defaults", {}), {})
        self.assertEqual(self.lifetime("oversight_failed"), [])
        self.assertEqual(self.lifetime("decision_default_check_failed"), [])
        with self.assertRaises(CostLimitReached):
            self.agent.oversight.evaluate(later)
        self.assertNotIn("attempt_at", read_json(self.paths.runtime / "oversight.json")["evaluation"])
        # After the reset the same checks run.
        self.clock.advance(5 * 3600)
        self.judge_engine.responses = ["VERDICT: apply\nREASON: reversible"]
        done = self.agent.oversight.process_defaults(later)
        self.assertEqual(done, [f"default for {self.agent.oversight.open_decisions()[0]['id']} applied"]
                         if self.agent.oversight.open_decisions() else done)
        self.assertEqual(len(self.judge_engine.requests), 1)

    def test_a_one_off_run_stops_at_the_limit_with_a_clear_error(self) -> None:
        self.spend(10.0)
        with self.assertRaises(CostLimitReached):
            self.agent.run_turn(trigger="manual")
        self.assertEqual(len(self.worker.requests), 0)

    def test_notifications_are_not_lost_when_a_request_is_refused(self) -> None:
        self.agent.notifications.create(type="external_event", source="owner", summary="KEEP ME")
        self.spend(10.0)
        with self.assertRaises(CostLimitReached):
            self.agent.run_turn(trigger="manual")
        self.assertTrue(self.agent.notifications.has_new())
        self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 1)


class PauseTests(RuntimeCase):
    """run_forever with the fake clock: the pause, the queue and the resume."""

    def run_loop(self, respond, *, max_polls: int = 400) -> None:
        self.calls: list = []
        polls = {"n": 0}

        def sleeper(seconds):
            polls["n"] += 1
            self.on_poll(polls["n"])
            if polls["n"] > max_polls:
                self.agent._stop = True

        def recording(messages):
            self.calls.append(messages)
            return respond(messages)

        self.worker.complete = recording
        with mock.patch("artificium.runtime.time.sleep", side_effect=sleeper):
            self.agent.run_forever(quiet=True)

    def on_poll(self, number: int) -> None:
        pass

    def reply(self, spend: float):
        def respond(messages):
            self.spend(spend)
            return EngineReply("<think>no action</think>", usage={"input_tokens": 1, "output_tokens": 1})
        return respond

    def test_at_the_limit_the_instance_sleeps_keeps_its_queue_and_resumes_by_itself(self) -> None:
        self.spend(9.5)
        requests_at_pause: list[int] = []

        def on_poll(number: int) -> None:
            state = read_json(self.paths.sleep_state, {})
            if state.get("reason") == "cost_limit" and state.get("active"):
                if not requests_at_pause:
                    # Paused: write to the instance; nothing may be sent or consumed.
                    self.agent.notifications.create(type="external_event", source="owner",
                                                    summary="QUEUED DURING THE PAUSE")
                    requests_at_pause.append(len(self.calls))
                    self.pause_state = read_json(self.paths.runtime_state)
                elif number > 3 and not hasattr(self, "resumed"):
                    self.assertEqual(len(self.calls), requests_at_pause[0])
                    self.assertTrue(self.agent.notifications.has_new())
                    self.resumed = True
                    self.clock.advance(5 * 3600)   # the reset at local midnight
            elif getattr(self, "resumed", False) and len(self.calls) > requests_at_pause[0]:
                self.agent._stop = True

        self.on_poll = on_poll
        self.run_loop(self.reply(1.0))
        # One request crossed the limit, then nothing more until the reset.
        self.assertEqual(requests_at_pause, [1])
        self.assertEqual(self.pause_state["status"], "cost_limited")
        self.assertEqual(self.pause_state["reset_at"], "2026-05-05 00:00")
        self.assertEqual(self.pause_state["timezone"], "Asia/Tokyo")
        reached = self.lifetime("cost_limit_reached")
        self.assertEqual(len(reached), 1)
        self.assertEqual(reached[0]["reset_at"], "2026-05-05 00:00")
        ended = [r for r in self.lifetime("sleep_ended") if r.get("reason") == "cost_limit_reset"]
        self.assertEqual(len(ended), 1)
        # After the reset the queued message reached the model.
        self.assertGreater(len(self.calls), 1)
        self.assertIn("QUEUED DURING THE PAUSE", json.dumps(self.calls[-1]))
        self.assertEqual(self.agent.cost.status()["day"], "2026-05-05")

    def test_messages_do_not_end_the_pause(self) -> None:
        self.spend(10.0)

        def on_poll(number: int) -> None:
            self.agent.notifications.create(type="external_event", source="owner", summary=f"m{number}")

        self.on_poll = on_poll
        self.run_loop(self.reply(0.0), max_polls=10)
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(len(self.judge_engine.requests), 0)
        self.assertEqual(len(list(self.paths.notifications_new.glob("*.json"))), 11)
        self.assertEqual(read_json(self.paths.sleep_state)["reason"], "cost_limit")

    def test_due_oversight_checks_wait_during_the_pause(self) -> None:
        self.spend(10.0)
        with mock.patch.object(self.agent.oversight, "tick",
                               wraps=self.agent.oversight.tick) as tick:
            self.on_poll = lambda number: None
            self.run_loop(self.reply(0.0), max_polls=5)
        tick.assert_not_called()

    def test_raising_the_limit_and_restarting_ends_the_pause(self) -> None:
        self.spend(10.0)
        self.run_loop(self.reply(0.0), max_polls=2)
        self.assertTrue(read_json(self.paths.sleep_state)["active"])
        self.agent.config = dataclasses.replace(self.agent.config, daily_cost_limit=50.0)
        self.agent.cost.config = self.agent.config
        self.assertFalse(self.agent._sleep_active())
        self.assertEqual(read_json(self.paths.sleep_state)["reason"], "cost_limit_lifted")

    def test_the_limit_crossed_mid_turn_ends_the_turn_without_a_failure(self) -> None:
        self.spend(9.5)

        def respond(messages):
            self.spend(1.0)
            return EngineReply("<output>step</output><tool_call>{\"name\": \"read_file\", \"arguments\": "
                               "{\"path\": \"mind/self.txt\"}}</tool_call>",
                               usage={"input_tokens": 1, "output_tokens": 1})

        self.on_poll = lambda n: setattr(self.agent, "_stop", n > 3) if read_json(
            self.paths.sleep_state, {}).get("reason") == "cost_limit" else None
        self.run_loop(respond, max_polls=50)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.lifetime("turn_failed"), [])
        self.assertEqual(len(self.lifetime("cost_limit_reached")), 1)
