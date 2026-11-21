"""Spend limits over a rolling five hours, the day, the week and the month."""

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
    import test_cost as base
except ImportError:  # run as tests.test_cost_periods
    from tests import test_cost as base

from artificium import cost, display
from artificium.cli import main
from artificium.config import Config, ConfigStore, SecretsStore
from artificium.cost import CostLedger, CostLimitReached
from artificium.filesystem import Paths, read_json
from artificium.operator import _cost_lines
from artificium.runtime import Artificium

UTC = ZoneInfo("UTC")


def at(*moment, zone=UTC) -> dt.datetime:
    return dt.datetime(*moment, tzinfo=zone)


class LedgerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        self.paths.ensure_layout()

    def ledger(self, clock, **changes) -> CostLedger:
        changes.setdefault("cost_day_timezone", "UTC")
        return CostLedger(self.paths, base._config(**changes), clock)

    @staticmethod
    def spend(ledger: CostLedger, amount: float | None, connection: str = "worker") -> None:
        ledger.record(connection, {"input_tokens": 1, "output_tokens": 1}, amount)

    @staticmethod
    def period(ledger: CostLedger, key: str) -> dict:
        return next(item for item in ledger.status()["periods"] if item["period"] == key)


class CalendarPeriodTests(LedgerCase):
    def test_the_week_counts_from_monday_and_resets_the_next_monday(self) -> None:
        clock = base.FakeClock(at(2026, 3, 1, 23, 0))             # a Sunday: last week
        ledger = self.ledger(clock, weekly_cost_limit=10.0)
        self.spend(ledger, 100.0)
        clock.now = at(2026, 3, 2, 9, 0).timestamp()              # Monday
        self.spend(ledger, 3.0)
        clock.now = at(2026, 3, 4, 12, 0).timestamp()             # Wednesday
        self.spend(ledger, 4.0)
        week = self.period(ledger, "week")
        self.assertEqual(week["spent"], 7.0)
        self.assertFalse(week["limited"])
        self.assertEqual(week["reset_at"], "2026-03-09 00:00")
        self.spend(ledger, 3.0)
        status = ledger.status()
        self.assertTrue(status["limited"])
        self.assertEqual((status["period"], status["limited_periods"]), ("week", ["week"]))
        self.assertEqual(status["reset_at"], "2026-03-09 00:00")
        clock.now = at(2026, 3, 9, 0, 0, 1).timestamp()
        self.assertEqual(self.period(ledger, "week")["spent"], 0.0)
        self.assertFalse(ledger.status()["limited"])

    def test_the_month_resets_on_the_first_and_rolls_over_the_year(self) -> None:
        clock = base.FakeClock(at(2026, 3, 31, 10, 0))
        ledger = self.ledger(clock, monthly_cost_limit=20.0)
        self.spend(ledger, 15.0)
        self.assertEqual(self.period(ledger, "month")["spent"], 15.0)
        self.spend(ledger, 5.0)
        self.assertEqual(ledger.status()["reset_at"], "2026-04-01 00:00")
        clock.now = at(2026, 4, 1, 0, 0, 1).timestamp()
        self.assertEqual(self.period(ledger, "month")["spent"], 0.0)
        december = base.FakeClock(at(2026, 12, 20, 8, 0))
        year_end = self.ledger(december, monthly_cost_limit=1.0)
        self.spend(year_end, 2.0)
        self.assertEqual(year_end.status()["reset_at"], "2027-01-01 00:00")

    def test_the_week_and_month_follow_the_configured_zone(self) -> None:
        tokyo = ZoneInfo("Asia/Tokyo")
        # Sunday 23:30 in Tokyo is already Sunday 14:30 UTC; Monday starts at Tokyo midnight.
        clock = base.FakeClock(at(2026, 3, 8, 23, 30, zone=tokyo))
        ledger = self.ledger(clock, weekly_cost_limit=5.0, cost_day_timezone="Asia/Tokyo")
        self.spend(ledger, 5.0)
        self.assertTrue(ledger.status()["limited"])
        self.assertEqual(ledger.status()["reset_at"], "2026-03-09 00:00")
        clock.now = at(2026, 3, 9, 0, 1, zone=tokyo).timestamp()
        self.assertFalse(ledger.status()["limited"])

    def test_without_limits_the_status_keeps_the_shape_it_always_had(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock)
        self.spend(ledger, 1.5)
        status = ledger.status()
        self.assertEqual((status["spent"], status["limit"], status["fraction"]), (1.5, None, None))
        self.assertEqual((status["limited"], status["warn_due"], status["period"]), (False, False, "day"))
        self.assertEqual(status["reset_at"], "2026-03-05 00:00")
        self.assertEqual([item["period"] for item in status["periods"]], ["5h", "day", "week", "month"])
        self.assertTrue(all(item["limit"] is None for item in status["periods"]))
        # Spend is still tallied for each period, so the report can show it.
        self.assertEqual(self.period(ledger, "month")["spent"], 1.5)


class RollingWindowTests(LedgerCase):
    def test_the_window_frees_up_when_enough_old_spend_has_left_it(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=5.0)
        self.spend(ledger, 2.0)
        clock.advance(3600)
        self.spend(ledger, 2.0)
        clock.advance(3600)
        self.spend(ledger, 1.5)                                    # 14:00, total 5.5
        status = ledger.status()
        self.assertTrue(status["limited"])
        self.assertEqual(status["period"], "5h")
        # Dropping the 12:00 spend leaves 3.5: that is when requests can resume
        # (5 hours later, rounded up to a whole minute).
        self.assertEqual(status["reset_at"], "2026-03-04 17:01")
        clock.now = at(2026, 3, 4, 16, 59, 59).timestamp()
        self.assertTrue(ledger.status()["limited"])
        clock.now = at(2026, 3, 4, 17, 1).timestamp()
        status = ledger.status()
        self.assertFalse(status["limited"])
        self.assertEqual(self.period(ledger, "5h")["spent"], 3.5)

    def test_it_may_take_more_than_the_oldest_spend_to_get_under_the_limit(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=5.0)
        for _ in range(4):
            self.spend(ledger, 2.0)
            clock.advance(600)                                     # 12:00 .. 12:30
        clock.now = at(2026, 3, 4, 13, 0).timestamp()              # total 8.0
        # The two oldest must leave (8 - 2 - 2 = 4 < 5): the second one, at 12:10.
        self.assertEqual(ledger.status()["reset_at"], "2026-03-04 17:11")

    def test_a_single_request_over_the_limit_holds_until_it_leaves_the_window(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=5.0)
        self.spend(ledger, 50.0)
        self.assertEqual(ledger.status()["reset_at"], "2026-03-04 17:01")

    def test_a_limit_set_later_counts_what_was_spent_just_before(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        self.spend(self.ledger(clock), 4.0)
        clock.advance(1800)
        self.assertTrue(self.ledger(clock, five_hour_cost_limit=3.0).status()["limited"])

    def test_the_recent_list_drops_old_spend_and_ignores_unknown_costs(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=5.0)
        self.spend(ledger, 1.0)
        self.spend(ledger, None)                                   # unknown: not counted
        self.spend(ledger, 0.0)                                    # free: nothing to remember
        self.assertEqual(len(read_json(ledger.path)["recent"]), 1)
        clock.advance(6 * 3600)
        self.spend(ledger, 2.0)
        self.assertEqual(len(read_json(ledger.path)["recent"]), 1)
        self.assertEqual(self.period(ledger, "5h")["spent"], 2.0)


class CombinedLimitTests(LedgerCase):
    def test_the_period_that_frees_up_last_decides_the_pause(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=3.0, daily_cost_limit=10.0)
        self.spend(ledger, 4.0)
        status = ledger.status()
        self.assertEqual((status["period"], status["limited_periods"]), ("5h", ["5h"]))
        self.assertEqual(status["reset_at"], "2026-03-04 17:01")
        self.spend(ledger, 7.0)                                    # the day is now at 11
        status = ledger.status()
        self.assertEqual(status["limited_periods"], ["5h", "day"])
        self.assertEqual((status["period"], status["reset_at"]), ("day", "2026-03-05 00:00"))

    def test_when_nothing_is_limited_the_nearest_to_its_limit_is_reported(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=10.0, weekly_cost_limit=20.0)
        self.spend(ledger, 4.0)
        status = ledger.status()
        self.assertFalse(status["limited"])
        self.assertEqual((status["period"], status["limit"], status["fraction"]), ("5h", 10.0, 0.4))

    def test_both_connections_count_toward_every_period(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, weekly_cost_limit=10.0, five_hour_cost_limit=10.0)
        self.spend(ledger, 6.0, "worker")
        self.spend(ledger, 4.0, "oversight")
        self.assertEqual(ledger.status()["limited_periods"], ["5h", "week"])


class NoticeTests(LedgerCase):
    def test_each_period_is_warned_once_per_occurrence(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))           # a Wednesday
        ledger = self.ledger(clock, five_hour_cost_limit=3.0, weekly_cost_limit=10.0,
                             monthly_cost_limit=100.0)
        self.spend(ledger, 2.5)
        due = [item["period"] for item in ledger.status()["periods"] if item["warn_due"]]
        self.assertEqual(due, ["5h"])
        ledger.mark_warned(["5h"])
        self.assertFalse(ledger.status()["warn_due"])
        # Five hours later the old spend is gone; new spend at 80% warns again.
        clock.advance(5 * 3600 + 60)
        self.spend(ledger, 2.5)
        self.assertEqual([i["period"] for i in ledger.status()["periods"] if i["warn_due"]], ["5h"])
        ledger.mark_warned(["5h"])
        # The week crosses 80% too, once; the next week it can warn again.
        self.spend(ledger, 3.6)                                     # the week is at 8.6 of 10
        self.assertEqual([i["period"] for i in ledger.status()["periods"] if i["warn_due"]], ["week"])
        ledger.mark_warned()                                        # default: every due period
        self.assertFalse(ledger.status()["warn_due"])
        clock.now = at(2026, 3, 9, 9, 0).timestamp()
        self.spend(ledger, 8.5)
        self.assertEqual([i["period"] for i in ledger.status()["periods"] if i["warn_due"]], ["week"])

    def test_the_monthly_and_daily_notices_follow_their_own_calendar(self) -> None:
        clock = base.FakeClock(at(2026, 3, 31, 9, 0))
        ledger = self.ledger(clock, monthly_cost_limit=10.0, daily_cost_limit=100.0)
        self.spend(ledger, 8.5)
        self.assertEqual([i["period"] for i in ledger.status()["periods"] if i["warn_due"]], ["month"])
        ledger.mark_warned(["month"])
        clock.now = at(2026, 4, 1, 9, 0).timestamp()
        self.assertFalse(ledger.status()["warn_due"])
        self.spend(ledger, 8.5)
        self.assertTrue(ledger.status()["warn_due"])

    def test_the_header_and_the_report_name_every_limit(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=3.0, daily_cost_limit=10.0,
                             weekly_cost_limit=50.0)
        self.spend(ledger, 2.4)
        line = cost.header_line(ledger.config, ledger.status(), True)
        self.assertEqual(line, "Spend vs limits (USD): 5h 2.40 of 3.00 (80%); "
                               "today 2.40 of 10.00 (24%); week 2.40 of 50.00 (5%)")
        self.spend(ledger, 1.0)
        report = cost.format_report(ledger.report())
        self.assertIn("5-hour limit: 3.00 USD for both connections together; spent 3.40 (113%); "
                      "REACHED, requests are held back", report)
        self.assertIn("Requests resume at 2026-03-04 17:01", report)
        self.assertIn("Daily limit: 10.00 USD", report)
        self.assertIn("Resets at 2026-03-05 00:00 (UTC)", report)
        self.assertIn("Weekly limit: 50.00 USD", report)
        self.assertIn("Resets at 2026-03-09 00:00 (UTC)", report)
        # The month has spend but no limit: it is still shown.
        self.assertIn("Monthly limit: none; spent this month: 3.40 USD", report)
        none = self.ledger(clock)
        self.assertIn("Daily limit: none", cost.format_report(none.report()))

    def test_the_status_line_for_the_operator(self) -> None:
        clock = base.FakeClock(at(2026, 3, 4, 12, 0))
        ledger = self.ledger(clock, five_hour_cost_limit=3.0, monthly_cost_limit=100.0)
        self.spend(ledger, 3.5)
        lines = _cost_lines(ledger.status())
        self.assertEqual(len(lines), 1)
        self.assertIn("Spend vs limits (USD): 5h 3.50 of 3.00 (117%) REACHED, held until "
                      "2026-03-04 17:01; month 3.50 of 100.00 (4%)", lines[0])

    def test_the_limits_in_force_are_remembered_between_starts(self) -> None:
        ledger = self.ledger(base.FakeClock(at(2026, 3, 4, 12, 0)))
        self.assertIsNone(ledger.note_limits({"5h": None, "day": 5.0}))
        self.assertEqual(ledger.note_limits({"5h": 1.0, "day": 5.0}), {"5h": None, "day": 5.0})


class ConfigurationTests(unittest.TestCase):
    FIELDS = ("five_hour_cost_limit", "daily_cost_limit", "weekly_cost_limit", "monthly_cost_limit")

    def test_every_limit_is_validated_the_same_way(self) -> None:
        for name in self.FIELDS:
            for bad in (0, -1, True, "5", float("nan")):
                with self.assertRaisesRegex(ValueError, name, msg=f"{name}={bad!r}"):
                    base._config(**{name: bad})
            self.assertEqual(getattr(base._config(**{name: 0.5}), name), 0.5)

    def test_all_four_limits_are_saved_and_a_file_without_them_means_no_limit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            paths = Paths(Path(folder))
            paths.ensure_layout()
            ConfigStore(paths).save(base._config(**base.PRICES, five_hour_cost_limit=2.0,
                                                 daily_cost_limit=10.0, weekly_cost_limit=40.0,
                                                 monthly_cost_limit=120.0))
            document = json.loads(paths.config.read_text())
            self.assertEqual({name: document["harness"][name] for name in self.FIELDS},
                             {"five_hour_cost_limit": 2.0, "daily_cost_limit": 10.0,
                              "weekly_cost_limit": 40.0, "monthly_cost_limit": 120.0})
            for name in ("five_hour_cost_limit", "weekly_cost_limit", "monthly_cost_limit"):
                del document["harness"][name]
            paths.config.write_text(json.dumps(document))
            loaded = ConfigStore(paths).load()
            self.assertEqual((loaded.five_hour_cost_limit, loaded.weekly_cost_limit,
                              loaded.monthly_cost_limit, loaded.daily_cost_limit),
                             (None, None, None, 10.0))

    def test_any_limit_needs_prices_and_the_message_says_so(self) -> None:
        for name in self.FIELDS:
            problem = cost.cost_limit_problem(base._config(**{name: 5.0}))
            self.assertIn("--price-input", problem, name)
            self.assertIn("cost limits", problem, name)
            self.assertIsNone(cost.cost_limit_problem(base._config(**base.PRICES, **{name: 5.0})))
        self.assertIsNone(cost.cost_limit_problem(base._config()))


class CliCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.paths = Paths(Path(self.temporary.name))
        base._seed_install(self.paths)
        ConfigStore(self.paths).save(base._config(**base.PRICES))
        patcher = mock.patch.dict(os.environ, {}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--root", str(self.paths.install), *argv])
        return code, out.getvalue(), err.getvalue()

    def test_the_four_limits_are_set_and_cleared_from_the_command_line(self) -> None:
        code, out, err = self.run_cli(
            "configure", "harness", "--five-hour-cost-limit", "3", "--daily-cost-limit", "10",
            "--weekly-cost-limit", "40", "--monthly-cost-limit", "120",
            "--cost-day-timezone", "Europe/Paris")
        self.assertEqual(code, 0, err)
        saved = ConfigStore(self.paths).load()
        self.assertEqual((saved.five_hour_cost_limit, saved.daily_cost_limit,
                          saved.weekly_cost_limit, saved.monthly_cost_limit), (3.0, 10.0, 40.0, 120.0))
        self.assertEqual(saved.price_input, 3.0)                   # nothing else was touched
        self.assertIn("5-hour cost limit: 3 USD, any rolling 5 hours", out)
        self.assertIn("Daily cost limit: 10 USD, periods end at midnight (Europe/Paris)", out)
        self.assertIn("Weekly cost limit: 40 USD", out)
        self.assertIn("Monthly cost limit: 120 USD", out)
        code, out, err = self.run_cli("configure", "harness", "--weekly-cost-limit", "off",
                                      "--five-hour-cost-limit", "off")
        self.assertEqual(code, 0, err)
        saved = ConfigStore(self.paths).load()
        self.assertEqual((saved.five_hour_cost_limit, saved.weekly_cost_limit,
                          saved.daily_cost_limit, saved.monthly_cost_limit), (None, None, 10.0, 120.0))
        self.run_cli("configure", "harness", "--daily-cost-limit", "off", "--monthly-cost-limit", "off")
        code, out, _ = self.run_cli("configure", "harness", "--cost-day-timezone", "local")
        self.assertIn("Cost limits: none", out)

    def test_a_limit_is_refused_on_a_connection_and_without_prices(self) -> None:
        self.assertEqual(self.run_cli("configure", "model", "--weekly-cost-limit", "1")[0], 2)
        ConfigStore(self.paths).save(base._config())
        code, _, err = self.run_cli("configure", "harness", "--monthly-cost-limit", "5")
        self.assertEqual(code, 2)
        self.assertIn("--price-input", err)
        self.assertIsNone(ConfigStore(self.paths).load().monthly_cost_limit)

    def test_the_interactive_menu_asks_for_each_limit(self) -> None:
        from artificium.setup import SetupOptions, SetupWizard
        answers = iter(["USD", "3", "0.3", "unset", "15",           # prices of the worker
                        "2", "10", "40", "off", "Europe/Paris", "y"])
        prompts: list[str] = []

        def answer(prompt: str = "") -> str:
            prompts.append(prompt)
            return next(answers)

        with mock.patch("builtins.input", side_effect=answer), contextlib.redirect_stdout(io.StringIO()):
            SetupWizard(self.paths).reconfigure(SetupOptions(scope="cost"), interactive=True)
        saved = ConfigStore(self.paths).load()
        self.assertEqual((saved.five_hour_cost_limit, saved.daily_cost_limit, saved.weekly_cost_limit,
                          saved.monthly_cost_limit, saved.cost_day_timezone),
                         (2.0, 10.0, 40.0, None, "Europe/Paris"))
        asked = " ".join(prompts)
        for text in ("5-hour spend limit", "Daily spend limit", "Weekly spend limit",
                     "Monthly spend limit"):
            self.assertIn(text, asked)

    def test_the_cost_command_shows_every_limit(self) -> None:
        ConfigStore(self.paths).save(base._config(**base.PRICES, five_hour_cost_limit=3.0,
                                                  weekly_cost_limit=40.0, cost_day_timezone="UTC"))
        ledger = CostLedger(self.paths, ConfigStore(self.paths).load())
        ledger.record("worker", {"input_tokens": 1, "output_tokens": 1}, 3.2)
        code, out, _ = self.run_cli("cost")
        self.assertEqual(code, 0)
        self.assertIn("5-hour limit: 3.00 USD", out)
        self.assertIn("REACHED", out)
        self.assertIn("Weekly limit: 40.00 USD", out)
        code, out, _ = self.run_cli("status")
        self.assertIn("Spend vs limits (USD): 5h 3.20 of 3.00 (107%) REACHED", out)


class FiveHourRuntimeTests(base.RuntimeCase):
    """The harness refuses requests at a period's limit, whichever period it is."""

    config_changes = {"five_hour_cost_limit": 3.0, "decision_review": False}

    def test_the_rolling_limit_stops_both_connections_and_names_itself(self) -> None:
        self.spend(2.0)
        self.spend(1.5, "oversight")
        with self.assertRaises(CostLimitReached) as caught:
            self.ask()
        message = str(caught.exception)
        self.assertIn("5-hour cost limit reached (3.50 of 3.00 USD)", message)
        self.assertIn("no request was sent", message)
        self.assertIn("Requests resume at 2026-05-05 01:01 (Asia/Tokyo)", message)
        with self.assertRaises(CostLimitReached):
            self.agent._oversight_complete([{"role": "user", "content": "judge"}])
        self.assertEqual((len(self.worker.requests), len(self.judge_engine.requests)), (0, 0))
        self.assertEqual(self.lifetime("model_request"), [])
        self.assertEqual(self.lifetime("oversight_connection_failed"), [])
        # The day is nowhere near its limit; only the window holds the requests.
        self.assertEqual(self.agent.cost.status()["limited_periods"], ["5h"])
        self.clock.advance(5 * 3600 + 120)
        self.ask()
        self.assertEqual(len(self.worker.requests), 1)

    def test_the_weekly_and_monthly_limits_are_enforced_too(self) -> None:
        for changes in ({"weekly_cost_limit": 5.0}, {"monthly_cost_limit": 5.0}):
            with self.subTest(changes):
                self.agent.config = dataclasses.replace(
                    self.agent.config, five_hour_cost_limit=None, daily_cost_limit=None, **changes)
                self.agent.cost.config = self.agent.config
                self.agent.cost.path.unlink(missing_ok=True)
                self.spend(5.0)
                with self.assertRaises(CostLimitReached) as caught:
                    self.ask()
                self.assertIn(("weekly" if "weekly_cost_limit" in changes else "monthly")
                              + " cost limit reached", str(caught.exception))
                self.assertEqual(len(self.worker.requests), 0)

    def test_the_pause_waits_for_the_period_that_frees_up_last(self) -> None:
        # 20:00 in Tokyo: the daily limit frees at midnight, the window at 01:01.
        self.spend(10.5)
        self.agent._start_cost_pause()
        state = read_json(self.paths.sleep_state)
        self.assertEqual(state["reason"], "cost_limit")
        window = dt.datetime(2026, 5, 5, 1, 1, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()
        self.assertEqual(state["wake_at_epoch"], window)
        reached = self.lifetime("cost_limit_reached")[-1]
        self.assertEqual((reached["period"], reached["period_label"]), ("5h", "5-hour"))
        self.assertEqual(reached["limited_periods"], ["5h", "day"])
        runtime = read_json(self.paths.runtime_state)
        self.assertEqual(runtime["status"], "cost_limited")
        self.assertEqual(runtime["reset_at"], "2026-05-05 01:01")
        # Late in the evening the order is the other way round: the day decides.
        self.clock.advance(3 * 3600 + 1800)                         # 23:30
        self.agent._start_cost_pause()
        self.assertEqual(read_json(self.paths.runtime_state)["period"], "5h")

    def test_a_pause_for_the_window_ends_when_the_window_frees_up(self) -> None:
        self.spend(3.5)
        self.agent._start_cost_pause()
        self.assertTrue(self.agent._sleep_active())
        self.clock.advance(4 * 3600)
        self.assertTrue(self.agent._sleep_active())
        self.clock.advance(3600 + 120)
        self.assertFalse(self.agent._sleep_active())
        self.assertEqual(read_json(self.paths.sleep_state)["reason"], "cost_limit_reset")
        self.assertFalse(self.agent._cost_limited())

    def test_the_notice_for_the_window_comes_once_at_eighty_percent(self) -> None:
        self.spend(2.3)
        self.assertEqual(self.agent._cost_events(), [])
        self.spend(0.2)
        events = self.agent._cost_events()
        self.assertEqual(len(events), 1)
        self.assertIn("5-HOUR COST LIMIT NEARING", events[0])
        self.assertIn("Spent in the last 5 hours: 2.50 of 3.00 USD (83%)", events[0])
        self.assertIn("0.50 USD remain", events[0])
        self.assertIn("until enough of the last 5 hours' spend has aged out", events[0])
        self.assertEqual(self.agent._cost_events(), [])
        warned = self.lifetime("cost_warning_issued")
        self.assertEqual([item["period"] for item in warned], ["5h"])

    def test_the_daily_notice_keeps_its_wording(self) -> None:
        self.agent.config = dataclasses.replace(self.agent.config, five_hour_cost_limit=None)
        self.agent.cost.config = self.agent.config
        self.spend(8.5)
        text = self.agent._cost_events()[0]
        self.assertIn("DAILY COST LIMIT NEARING", text)
        self.assertIn("Spent today: 8.50 of 10.00 USD (85%)", text)
        self.assertIn("stop until 2026-05-05 00:00 (Asia/Tokyo)", text)

    def test_the_header_lists_the_limits_in_force(self) -> None:
        self.spend(1.5)
        self.assertIn("Spend vs limits (USD): 5h 1.50 of 3.00 (50%); today 1.50 of 10.00 (15%)\n",
                      self.agent._state_header("test", 1_000))


class LimitsInForceTests(base.RuntimeCase):
    """A raised or removed limit is noticed at the next start."""

    config_changes = {"five_hour_cost_limit": 3.0, "weekly_cost_limit": 20.0,
                      "decision_review": False}

    def restart(self, **changes) -> Artificium:
        store = ConfigStore(self.paths)
        store.save(dataclasses.replace(store.load(), **changes))
        console = mock.Mock()
        self.console = console
        return Artificium(self.paths, engine=self.worker, console=console)

    def test_the_limits_are_recorded_at_every_start(self) -> None:
        record = self.lifetime("cost_limits_in_force")[-1]
        self.assertEqual(record["limits"], {"5h": 3.0, "day": 10.0, "week": 20.0, "month": None})
        self.assertEqual(record["timezone"], "Asia/Tokyo")
        self.assertEqual(self.lifetime("cost_limits_loosened"), [])

    def test_raising_or_removing_a_limit_is_logged_and_shown(self) -> None:
        self.restart(five_hour_cost_limit=6.0, weekly_cost_limit=None)
        loosened = self.lifetime("cost_limits_loosened")
        self.assertEqual(len(loosened), 1)
        self.assertEqual(loosened[0]["changes"], {"5h": {"before": 3.0, "after": 6.0},
                                                  "week": {"before": 20.0, "after": None}})
        shown = " ".join(str(call) for call in self.console.line.call_args_list)
        self.assertIn("cost limits loosened since the last start", shown)
        self.assertIn("week 20 -> none", shown)

    def test_tightening_or_adding_a_limit_is_not_flagged(self) -> None:
        self.restart(five_hour_cost_limit=2.0, monthly_cost_limit=50.0)
        self.assertEqual(self.lifetime("cost_limits_loosened"), [])
        self.assertEqual(self.lifetime("cost_limits_in_force")[-1]["limits"]["month"], 50.0)


class FiveHourLoopTests(base.RuntimeCase):
    """run_forever with a fake clock: pause at the rolling limit, queue, resume."""

    config_changes = {"five_hour_cost_limit": 3.0, "decision_review": False}

    def test_the_loop_pauses_at_the_window_keeps_its_queue_and_resumes(self) -> None:
        self.spend(2.5)
        calls: list = []
        seen = {"paused_calls": None, "resumed": False}

        def respond(messages):
            calls.append(messages)
            self.spend(1.0)
            return base.EngineReply("<think>no action</think>",
                                    usage={"input_tokens": 1, "output_tokens": 1})

        def sleeper(seconds):
            state = read_json(self.paths.sleep_state, {})
            if state.get("reason") == "cost_limit" and state.get("active"):
                if seen["paused_calls"] is None:
                    seen["paused_calls"] = len(calls)
                    self.paused = read_json(self.paths.runtime_state)
                    self.agent.notifications.create(type="external_event", source="owner",
                                                    summary="QUEUED DURING THE PAUSE")
                else:
                    self.assertEqual(len(calls), seen["paused_calls"])    # nothing is sent
                    self.assertTrue(self.agent.notifications.has_new())    # nothing is consumed
                    seen["resumed"] = True
                    self.clock.advance(5 * 3600 + 120)
            elif seen["resumed"] and len(calls) > seen["paused_calls"]:
                self.agent._stop = True

        self.worker.complete = respond
        with mock.patch("artificium.runtime.time.sleep", side_effect=sleeper):
            self.agent.run_forever(quiet=True)
        self.assertEqual(seen["paused_calls"], 1)                  # one request crossed the limit
        self.assertEqual(self.paused["status"], "cost_limited")
        self.assertEqual((self.paused["period"], self.paused["reset_at"]), ("5h", "2026-05-05 01:01"))
        reached = self.lifetime("cost_limit_reached")
        self.assertEqual([item["period"] for item in reached], ["5h"])
        ended = [r for r in self.lifetime("sleep_ended") if r.get("reason") == "cost_limit_reset"]
        self.assertEqual(len(ended), 1)
        self.assertIn("QUEUED DURING THE PAUSE", json.dumps(calls[-1]))
        self.assertFalse(self.agent._cost_limited())


class DisplayTests(unittest.TestCase):
    def test_the_watch_view_names_the_period(self) -> None:
        options = mock.Mock(width=100)
        line = display.render_cost_limit_reached(
            {"spent": 3.5, "limit": 3.0, "currency": "USD", "reset_at": "2026-03-04 17:01",
             "timezone": "UTC", "period_label": "5-hour"}, options)[0]
        self.assertIn("5-hour limit reached (3.5 of 3.0 USD)", line)
        old = display.render_cost_limit_reached(
            {"spent": 10, "limit": 10, "currency": "USD", "reset_at": "x", "timezone": "UTC"}, options)[0]
        self.assertIn("daily limit reached", old)
        text = display.render_cost_limits_loosened(
            {"changes": {"day": {"before": 10.0, "after": None}}}, options)[0]
        self.assertIn("day 10.0 -> none", text)


if __name__ == "__main__":
    unittest.main()
