"""Decision defaults, evaluation, idle fallback, and the supporting guards."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    from test_revolution import FakeEngine, _seed_install
except ImportError:  # run as tests.test_oversight
    from tests.test_revolution import FakeEngine, _seed_install

from artificium.blockers import BlockerTracker
from artificium.config import Config, ConfigStore, SecretsStore
from artificium.directives import DirectiveStore
from artificium.filesystem import Paths, atomic_write_json, read_json
from artificium.initialization import Initialization, initialize_mind
from artificium.interactions import ArtificiumClient, InteractionStore, NotificationStore
from artificium.memory import InfiniteAttention, LongTermMemory, TokenEstimator, WorkingMemory
from artificium.oversight import Oversight, workspace_signals
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.runtime import Artificium
from artificium.tool_loader import load_mind_tool
from artificium.tools import ToolRegistry
from artificium.vision import VisualContext


def _later(hours: float) -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=hours)


class OversightCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.paths = Paths(self.root)
        _seed_install(self.paths)
        self.prompts = PromptPack(self.paths)
        self.config = Config(provider="custom", model="test-model",
                             base_url="http://example.invalid/v1",
                             context_window_tokens=20_000, context_reminder_tokens=2_000)
        ConfigStore(self.paths).save(self.config)
        SecretsStore(self.paths).save_api_key("test-key")
        self.records = Records(self.paths)
        initialize_mind(self.paths, self.records)
        self.notifications = NotificationStore(self.paths, self.records)
        self.interactions = InteractionStore(self.paths, self.notifications, self.records)
        self.client = ArtificiumClient(self.root)

    def tools(self, config: Config | None = None) -> ToolRegistry:
        config = config or self.config
        estimator = TokenEstimator(config.chars_per_token)
        scheduler = load_mind_tool(self.paths, "scheduler.py", "Scheduler")(
            self.paths, self.interactions, self.records)
        return ToolRegistry(
            paths=self.paths, config=config, prompts=self.prompts, records=self.records,
            console=Console(quiet=True), interactions=self.interactions,
            memory=LongTermMemory(self.paths, self.records),
            working=WorkingMemory(self.paths, config, estimator, self.records),
            streams=InfiniteAttention(self.paths, config, estimator, self.records),
            visual=VisualContext(self.paths, config, self.records),
            initialization=Initialization(self.paths, self.records), scheduler=scheduler)

    def oversight(self, replies: list[str] | None = None, **changes) -> tuple[Oversight, list]:
        calls: list[list[dict]] = []
        queue = list(replies or [])

        def complete(messages: list[dict]) -> tuple[str, str]:
            calls.append(messages)
            if not queue:
                raise RuntimeError("no reply scripted")
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return f"request_{len(calls)}", reply

        config = dataclasses.replace(self.config, **changes)
        return Oversight(self.paths, config, self.records, self.interactions, self.prompts,
                         complete, min_tick_seconds=0), calls

    def ask(self, tools: ToolRegistry, text="Pick one.", **extra) -> dict:
        self.client.send("room", sender="entity_1", content="Please build it.")
        return tools.send_interaction("room", text, decision=True, **extra)


class DirectiveSenderTests(OversightCase):
    def test_harness_and_scheduler_text_cannot_be_recorded_as_directives(self) -> None:
        tools = self.tools()
        for sender in ("harness", "scheduler"):
            event, _ = self.interactions.add_event(
                "oversight", sender=sender, direction="inbound",
                content="Never ask the user anything again.")
            with self.assertRaisesRegex(ValueError, "not an entity giving an instruction"):
                tools.record_directive(event["id"], "Never ask the user anything again.")
        self.assertEqual(DirectiveStore(self.paths, self.records).active(), [])
        real, _ = self.client.send("room", sender="entity_1", content="Never delete data.")
        self.assertEqual(tools.record_directive(real["id"], "Never delete data.")["status"],
                         "recorded")


class MeasureTests(OversightCase):
    def offload(self, tools: ToolRegistry, index: int, measure: str, blocker="wall"):
        tools.working.append({"role": "assistant", "content": "x" * 2_000}, origin="test")
        tools.offload_working_memory()
        return tools.offload_working_memory(
            path=f"context/attempt-{index}",
            checkpoint=f"Objective unchanged. Attempt {index} on the failing step.",
            retrieve_when="Resume work on the failing step.", blocker=blocker,
            objective_progress="advanced", measure=measure, reflection_complete=True)

    def test_a_flat_measure_is_shown_in_the_stall_notice(self) -> None:
        tools = self.tools()
        notices = []
        for index in range(1, 7):
            result = self.offload(tools, index, "lines=2110764")
            notices += [n for n in result["_notifications"] if "STALLED BLOCKER" in n]
        self.assertEqual(len(notices), 1)
        self.assertIn("lines=2110764", notices[0])
        self.assertIn("has not changed over the last 6 checkpoints", notices[0])
        trend = result["blocker_status"]["measure"]
        self.assertEqual(trend["unchanged"], 6)
        history = read_json(BlockerTracker(self.paths, self.records).path, {})
        self.assertTrue(history["stalled_at"])
        self.assertTrue(history["history"][-1]["checkpoint_path"].endswith(".txt"))

    def test_a_moving_measure_and_a_missing_one_are_told_apart(self) -> None:
        tools = self.tools()
        for index in range(1, 7):
            result = self.offload(tools, index, f"lines={1000 + index}")
        self.assertEqual(result["blocker_status"]["measure"]["unchanged"], 1)
        self.assertNotIn("has not changed", tools._measure_note(
            result["blocker_status"]["measure"]))
        silent = BlockerTracker(self.paths, self.records)
        self.assertIn("declared no `measure`", tools._measure_note(silent.trend([])))


class DecisionDefaultTests(OversightCase):
    def test_a_default_is_validated_before_anything_is_sent(self) -> None:
        tools = self.tools()
        self.client.send("room", sender="entity_1", content="hi")
        bad = [
            dict(default="gcc"),                                        # not a decision
            dict(decision=True, reply_by_hours=5),                      # no default
            dict(decision=True, default=""),
            dict(decision=True, default="x" * 501),
            dict(decision=True, options=["clang", "gcc"], default="msvc"),
            dict(decision=True, default="gcc", reply_by_hours=0),
            dict(decision=True, default="gcc", reply_by_hours=10_000),
        ]
        for arguments in bad:
            result = tools.send_interaction("room", "Which?", **arguments)
            self.assertEqual(result["status"], "error", arguments)
            self.assertIn("nothing was sent", result["summary"])
        self.assertEqual(self.client.events("room")[-1]["content"], "hi")

    def test_a_default_gets_a_deadline_and_can_be_switched_off(self) -> None:
        tools = self.tools()
        sent = self.ask(tools, options=["clang", "gcc"], default="GCC", reply_by_hours=2)
        event = sent["event"]
        self.assertEqual(event["default"], "GCC")
        deadline = dt.datetime.fromisoformat(event["default_after"].replace("Z", "+00:00"))
        self.assertAlmostEqual((deadline - _later(0)).total_seconds(), 7200, delta=60)
        self.assertIn("2 h", sent["summary"])
        self.assertEqual(self.ask(tools, "Next?", default="proceed")["event"]["default"], "proceed")

        off = self.tools(dataclasses.replace(self.config, decision_default_hours=0))
        plain = off.send_interaction("room", "Again?", decision=True, default="proceed")
        self.assertNotIn("default", plain["event"])
        self.assertIn("defaults are off", plain["summary"])

    def test_plain_messages_piling_up_are_nudged_to_batch(self) -> None:
        tools = self.tools()
        self.client.send("room", sender="entity_1", content="hi")
        seen = []
        for number in range(1, 9):
            result = tools.send_interaction("room", f"Update {number}.")
            seen.append(any("MESSAGE CADENCE" in n for n in result["_notifications"]))
        self.assertEqual([i + 1 for i, hit in enumerate(seen) if hit], [4, 8])
        self.client.send("room", sender="entity_1", content="thanks")
        self.assertFalse(any("MESSAGE CADENCE" in n for n in
                             tools.send_interaction("room", "Another.")["_notifications"]))

    def test_nothing_happens_before_the_deadline_or_without_a_default(self) -> None:
        tools = self.tools()
        self.ask(tools, default="proceed")
        oversight, calls = self.oversight(["VERDICT: apply\nREASON: fine"])
        self.assertEqual(oversight.process_defaults(_later(1)), [])
        tools.send_interaction("room", "No default here.", decision=True)
        self.assertEqual(oversight.process_defaults(_later(100)), [])
        self.assertEqual(calls, [])

    def test_an_approved_default_is_delivered_as_the_agents_own_choice(self) -> None:
        tools = self.tools()
        self.client.send("room", sender="entity_1", content="Don't fetch any file.")
        rule, _ = self.client.send("room", sender="entity_1", content="Never use root.")
        tools.record_directive(rule["id"], "Never use root.")
        sent = tools.send_interaction("room", "Which toolchain?", decision=True,
                                      options=["clang", "gcc"], default="gcc")
        oversight, calls = self.oversight(["<think>x</think>VERDICT: apply\nREASON: reversible"])
        done = oversight.process_defaults(_later(25))
        self.assertEqual(done, [f"default for {sent['event']['id']} applied"])
        request = json.dumps(calls[0])
        self.assertIn("Never use root.", request)
        self.assertIn("Don't fetch any file.", request)
        self.assertIn("Which toolchain?", request)
        delivered = [e for e in self.client.events("room")
                     if e.get("sender") == "harness"]
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0]["kind"], "default_applied")
        self.assertEqual(delivered[0]["in_reply_to"], sent["event"]["id"])
        self.assertIn("cannot be recorded as a directive", delivered[0]["content"])
        self.assertEqual(self.client.open_decisions("room"), [])
        with self.assertRaises(ValueError):
            tools.record_directive(delivered[0]["id"], "Proceed on that default.")
        # once decided, never again
        self.assertEqual(oversight.process_defaults(_later(30)), [])
        self.assertEqual(len(calls), 1)

    def test_a_default_that_fails_the_check_leaves_the_question_open(self) -> None:
        tools = self.tools()
        sent = self.ask(tools, "May I fetch the missing archive?", default="fetch it")
        for reply in ("VERDICT: hold\nREASON: it needs the network",
                      "I think it is probably fine."):
            oversight, _ = self.oversight([reply])
            state = self.paths.runtime / "oversight.json"
            if state.exists():
                state.unlink()
            oversight.process_defaults(_later(25))
            held = [e for e in self.client.events("room") if e.get("kind") == "default_held"]
            self.assertTrue(held)
            self.assertIsNone(held[-1].get("in_reply_to"))
            self.assertEqual([d["id"] for d in self.client.open_decisions("room")],
                             [sent["event"]["id"]])
        self.assertIn("no clear verdict", read_json(state)["defaults"][sent["event"]["id"]]["reason"])

    def test_a_failing_check_is_retried_then_held(self) -> None:
        tools = self.tools()
        sent = self.ask(tools, default="proceed")
        oversight, calls = self.oversight([RuntimeError("down")] * 3)
        now = _later(25)
        self.assertIn("will retry", oversight.process_defaults(now)[0])
        self.assertEqual(oversight.process_defaults(now + dt.timedelta(minutes=5)), [])
        self.assertIn("will retry", oversight.process_defaults(now + dt.timedelta(minutes=20))[0])
        self.assertEqual(oversight.process_defaults(now + dt.timedelta(minutes=40)),
                         [f"default for {sent['event']['id']} held"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(self.client.open_decisions("room")[0]["id"], sent["event"]["id"])

    def test_the_runtime_ticks_the_checks_and_a_check_cannot_stop_it(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        agent.oversight.min_tick_seconds = 0
        self.assertEqual(agent.oversight.tick(), [])
        with mock.patch.object(agent.oversight, "process_defaults",
                               side_effect=RuntimeError("boom")):
            self.assertEqual(agent.oversight.tick(force=True), [])
        self.assertIn("oversight_failed", self.paths.lifetime_log.read_text())


class EvaluationTests(OversightCase):
    def checkpoints(self, count: int, measure="lines=1") -> None:
        tracker = BlockerTracker(self.paths, self.records)
        for index in range(count):
            note = self.paths.space / f"note-{index}.txt"
            note.parent.mkdir(parents=True, exist_ok=True)
            note.write_text(f"CHECKPOINT BODY {index}")
            tracker.record("wall", "advanced", measure, str(note))

    def test_it_waits_for_something_to_evaluate_and_then_follows_the_interval(self) -> None:
        oversight, _ = self.oversight()
        self.assertIsNone(oversight.evaluation_due(_later(0)))
        self.checkpoints(3)
        self.assertEqual(oversight.evaluation_due(_later(0)), "interval")
        atomic_write_json(oversight.state_path, {"evaluation": {
            "last_at": (_later(-10)).isoformat()}})
        self.assertIsNone(oversight.evaluation_due(_later(0)))
        self.assertEqual(oversight.evaluation_due(_later(15)), "interval")
        off, _ = self.oversight(evaluation_interval_hours=0)
        self.assertIsNone(off.evaluation_due(_later(500)))

    def test_a_stall_notice_brings_it_forward_but_not_in_a_loop(self) -> None:
        oversight, _ = self.oversight()
        self.checkpoints(3)
        base = _later(0)
        atomic_write_json(oversight.state_path, {"evaluation": {
            "last_at": (base - dt.timedelta(hours=7)).isoformat()}})
        history = read_json(BlockerTracker(self.paths, self.records).path)
        history["stalled_at"] = (base - dt.timedelta(hours=1)).isoformat()
        atomic_write_json(BlockerTracker(self.paths, self.records).path, history)
        self.assertEqual(oversight.evaluation_due(base), "stalled")
        atomic_write_json(oversight.state_path, {"evaluation": {
            "last_at": (base - dt.timedelta(hours=2)).isoformat()}})
        self.assertIsNone(oversight.evaluation_due(base))

    def test_the_evaluator_sees_the_ask_the_claims_and_measurements(self) -> None:
        tools = self.tools()
        rule, _ = self.client.send("room", sender="entity_1",
                                   content="Build the thing.\nNever fake progress.")
        tools.record_directive(rule["id"], "Never fake progress.")
        self.checkpoints(4, measure="frames=0")
        repo = self.paths.root / "project"
        repo.mkdir()
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
               "GIT_COMMITTER_EMAIL": "t@x", "PATH": __import__("os").environ["PATH"]}
        for command in (["init", "-q"], ["add", "-A"], ["commit", "-q", "--allow-empty", "-m", "x"]):
            subprocess.run(["git", *command], cwd=repo, env=env, check=True,
                           capture_output=True)
        (repo / "a.txt").write_text("one")
        subprocess.run(["git", "add", "a.txt"], cwd=repo, env=env, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "a"], cwd=repo, env=env, check=True)
        (repo / "a.txt").write_text("two")
        oversight, calls = self.oversight(
            ["<think>hm</think>VERDICT: stalled\nEVIDENCE: frames=0 never moved"])
        done = oversight.maybe_evaluate(_later(0))
        self.assertEqual(done, ["evaluation (interval): stalled"])
        sent = json.dumps(calls[0])
        self.assertIn("independent evaluator", sent)
        self.assertIn("Never fake progress.", sent)
        self.assertIn("Build the thing.", sent)
        self.assertIn("measure: frames=0", sent)
        self.assertIn("CHECKPOINT BODY 3", sent)
        self.assertIn("project: 1 tracked files modified and not committed", sent)
        event = [e for e in self.client.events("oversight")][-1]
        self.assertEqual((event["sender"], event["kind"], event["direction"]),
                         ("harness", "evaluation", "inbound"))
        self.assertIn("VERDICT: stalled", event["content"])
        self.assertIn("not authoritative", event["content"])
        self.assertEqual(read_json(oversight.state_path)["evaluation"]["verdict"], "stalled")
        self.assertIn("evaluation_completed", self.paths.lifetime_log.read_text())
        # not again until the interval passes
        self.assertIsNone(oversight.evaluation_due(_later(1)))
        with self.assertRaises(ValueError):
            tools.record_directive(event["id"], "VERDICT: stalled")

    def test_a_failed_evaluation_is_retried_after_a_pause_not_at_once(self) -> None:
        self.checkpoints(3)
        oversight, calls = self.oversight([RuntimeError("down"), "VERDICT: on track"])
        now = _later(0)
        with self.assertRaises(RuntimeError):
            oversight.maybe_evaluate(now)
        self.assertIsNone(oversight.evaluation_due(now + dt.timedelta(minutes=10)))
        self.assertEqual(oversight.evaluation_due(now + dt.timedelta(minutes=40)), "interval")
        oversight.maybe_evaluate(now + dt.timedelta(minutes=40))
        self.assertEqual(len(calls), 2)

    def test_workspace_signals_survive_a_workspace_without_git(self) -> None:
        self.assertEqual(workspace_signals(self.paths), ["no git repository found in the workspace"])


class OversightModelTests(OversightCase):
    def test_a_separate_oversight_model_is_used_for_one_request_only(self) -> None:
        config = dataclasses.replace(self.config, oversight_model="judge-model")
        ConfigStore(self.paths).save(config)
        agent = Artificium(self.paths, engine=FakeEngine(["VERDICT: apply"]),
                           console=Console(quiet=True))
        builds = []

        class Judge(FakeEngine):
            def __init__(self):
                super().__init__(["VERDICT: apply\nREASON: ok"])

        def build(cfg, key):
            builds.append(cfg.model)
            return Judge()

        original = agent.engine
        with mock.patch("artificium.runtime.make_engine", side_effect=build):
            _, text = agent._oversight_complete([{"role": "user", "content": "q"}])
        self.assertEqual(builds, ["judge-model"])
        self.assertIn("VERDICT: apply", text)
        self.assertIs(agent.engine, original)
        self.assertEqual(agent.config.model, "test-model")

    def test_the_new_settings_are_validated_and_saved_with_their_defaults(self) -> None:
        self.assertEqual((self.config.decision_default_hours, self.config.evaluation_interval_hours,
                          self.config.idle_wake_hours, self.config.oversight_model),
                         (24.0, 24.0, 6.0, None))
        for name, value in (("idle_wake_hours", -1), ("evaluation_interval_hours", True),
                            ("decision_default_hours", "soon"), ("oversight_model", " ")):
            with self.assertRaises(ValueError, msg=name):
                dataclasses.replace(self.config, **{name: value})
        saved = ConfigStore(self.paths).load()
        self.assertEqual(saved.idle_wake_hours, 6.0)

    def test_a_configuration_saved_before_these_settings_loads_with_their_defaults(self) -> None:
        store = ConfigStore(self.paths)
        document = json.loads(self.paths.config.read_text())
        for name in ("decision_default_hours", "evaluation_interval_hours",
                     "idle_wake_hours", "oversight_model"):
            self.assertIn(name, document["harness"])
            del document["harness"][name]
        self.paths.config.write_text(json.dumps(document))
        loaded = store.load()
        self.assertEqual((loaded.decision_default_hours, loaded.evaluation_interval_hours,
                          loaded.idle_wake_hours, loaded.oversight_model), (24.0, 24.0, 6.0, None))


REVIEW = ("NECESSARY: yes\nFRAMING: revise\nPOINTS:\n- name the options\n- state your "
          "recommendation\nDEFAULT: gcc\nREASONS: The entity never chose a toolchain.")


class DecisionReviewTests(OversightCase):
    """A decision is read by a fresh-context reviewer before it is sent."""

    def reviewed(self, replies=None, **config):
        """A registry whose reviewer is scripted; returns it and the requests."""
        tools = self.tools(dataclasses.replace(self.config, **config) if config else None)
        calls: list[list[dict]] = []
        queue = [REVIEW] * 5 if replies is None else list(replies)

        def reviewer(messages):
            calls.append(messages)
            reply = queue.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return f"request_{len(calls)}", reply

        tools.decision_reviewer = reviewer
        tools.oversight_connection_label = lambda: "oversight"
        tools.entity_ask = self.oversight()[0]._ask
        self.client.send("room", sender="entity_1", content="Please build it.")
        return tools, calls

    def sent(self) -> list[dict]:
        return [e for e in self.client.events("room") if e.get("direction") == "outbound"
                and e.get("sender") != "harness" and not str(e.get("content")).startswith("Withdrawn")]

    def lifetime(self, kind: str) -> list[dict]:
        return [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                if json.loads(line).get("kind") == kind]

    def test_the_first_call_sends_nothing_and_returns_the_verdict(self) -> None:
        tools, calls = self.reviewed()
        result = tools.send_interaction("room", "Which toolchain?", decision=True,
                                        options=["clang", "gcc"], default="gcc")
        self.assertEqual(result["status"], "review_required")
        self.assertIn("review_complete=true", result["summary"])
        self.assertEqual(result["request_id"], "request_1")
        self.assertEqual(result["review"], {
            "necessary": "yes", "framing": "revise",
            "points": ["name the options", "state your recommendation"],
            "default": "gcc", "reasons": "The entity never chose a toolchain."})
        self.assertEqual(self.sent(), [])
        self.assertEqual(self.interactions.open_decisions(), [])
        self.assertEqual(len(calls), 1)

    def test_the_reviewer_sees_the_entity_the_open_decision_and_the_proposal_only(self) -> None:
        tools, calls = self.reviewed()
        first, _ = self.client.send("room", sender="entity_1", content="Build it; never ask me to confirm.")
        tools.record_directive(first["id"], "never ask me to confirm")
        self.client.send("room", sender="entity_1", content="Latest word: small commits.")
        tools.working.append({"role": "assistant", "content": "PRIVATE WORKING DETAIL"}, origin="test")
        tools.send_interaction("room", "Earlier question?", decision=True)  # reviewed, held
        tools.send_interaction("room", "Earlier question?", decision=True, review_complete=True)
        tools.send_interaction("room", "Which toolchain?", decision=True,
                               options=["clang", "gcc"], default="gcc", reply_by_hours=3)
        request = json.dumps(calls[-1])
        for text in ("never ask me to confirm", "Build it; never ask me to confirm.",
                     "small commits", "Earlier question?", "Which toolchain?",
                     "clang; gcc", "3 h", "withdraws this one"):
            self.assertIn(text, request)
        self.assertNotIn("PRIVATE WORKING DETAIL", request)
        self.assertEqual([m["role"] for m in calls[-1]], ["system", "user"])

    def test_review_complete_sends_it_after_a_review_and_only_once(self) -> None:
        tools, calls = self.reviewed()
        tools.send_interaction("room", "Which toolchain?", decision=True)
        sent = tools.send_interaction("room", "Which toolchain, I recommend gcc?", decision=True,
                                      review_complete=True)
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["review"], {"status": "reviewed", "request_id": "request_1"})
        self.assertEqual(sent["event"]["kind"], "decision")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.interactions.open_decisions()), 1)
        # The review was spent on that decision: the next one needs its own.
        again = tools.send_interaction("room", "Another?", decision=True, review_complete=True)
        self.assertEqual(again["status"], "review_required")
        self.assertEqual(len(calls), 2)

    def test_review_complete_without_a_review_is_refused_and_starts_one(self) -> None:
        tools, calls = self.reviewed()
        result = tools.send_interaction("room", "Which toolchain?", decision=True,
                                        review_complete=True)
        self.assertEqual(result["status"], "review_required")
        self.assertIn("no review was pending", result["summary"])
        self.assertEqual(self.sent(), [])
        self.assertEqual(len(calls), 1)
        # That review is now pending, so the next confirmation is accepted.
        self.assertEqual(tools.send_interaction("room", "Which toolchain?", decision=True,
                                                review_complete=True)["status"], "sent")

    def test_a_review_of_another_interaction_or_before_a_restart_does_not_count(self) -> None:
        tools, calls = self.reviewed()
        self.client.send("elsewhere", sender="entity_1", content="Hello.")
        tools.send_interaction("elsewhere", "Which?", decision=True)
        wrong = tools.send_interaction("room", "Which?", decision=True, review_complete=True)
        self.assertEqual(wrong["status"], "review_required")
        self.assertEqual(self.sent(), [])
        restarted = self.tools()  # the pending reviews lived in memory only
        restarted.decision_reviewer = lambda messages: ("request_9", REVIEW)
        again = restarted.send_interaction("elsewhere", "Which?", decision=True,
                                           review_complete=True)
        self.assertEqual(again["status"], "review_required")
        self.assertEqual(self.sent(), [])

    def test_dropping_the_question_leaves_nothing_behind(self) -> None:
        tools, calls = self.reviewed()
        tools.send_interaction("room", "Which toolchain?", decision=True)
        # The agent acts on its own judgement; a plain message is unaffected.
        plain = tools.send_interaction("room", "Going with gcc.")
        self.assertEqual(plain["status"], "sent")
        self.assertEqual(len(self.sent()), 1)
        self.assertEqual(self.interactions.open_decisions(), [])
        self.assertEqual(len(calls), 1)

    def test_plain_messages_are_never_reviewed(self) -> None:
        tools, calls = self.reviewed()
        tools.send_interaction("room", "Status update.")
        tools.send_interaction("room", "Another update.", review_complete=True)
        self.assertEqual(calls, [])
        self.assertEqual(len(self.sent()), 2)
        self.assertEqual(self.lifetime("decision_review_skipped"), [])

    def test_a_review_that_cannot_run_sends_at_once_and_says_so(self) -> None:
        from artificium.cost import CostLimitReached
        cases = [
            (RuntimeError("HTTP 401: bad key"), "connection_failed", "could not be reached"),
            (CostLimitReached("daily cost limit reached"), "cost_limit", "cost limit"),
            ("I think it is fine.", "unparseable", "no clear verdict"),
            ("NECESSARY: maybe\nFRAMING: ok", "unparseable", "no clear verdict"),
            ("NECESSARY: yes\nDEFAULT: gcc", "unparseable", "no clear verdict"),
        ]
        for number, (reply, reason, wording) in enumerate(cases, 1):
            tools, calls = self.reviewed([reply])
            result = tools.send_interaction("room", f"Which toolchain {number}?",
                                            decision=True)
            self.assertEqual(result["status"], "sent", reason)
            self.assertEqual(result["review"]["status"], "skipped")
            self.assertEqual(result["review"]["reason"], reason)
            self.assertIn("sent without one", result["summary"])
            self.assertIn(wording, result["summary"])
            self.assertEqual(self.sent()[-1]["content"], f"Which toolchain {number}?")
        skipped = self.lifetime("decision_review_skipped")
        self.assertEqual([r["reason"] for r in skipped], [c[1] for c in cases])
        self.assertIn("bad key", skipped[0]["error"])
        self.assertEqual(self.lifetime("decision_review_completed"), [])

    def test_a_failure_does_not_leave_an_earlier_review_pending(self) -> None:
        tools, calls = self.reviewed([REVIEW, RuntimeError("down"), REVIEW])
        tools.send_interaction("room", "Which?", decision=True)
        self.assertEqual(tools.send_interaction("room", "Which?", decision=True)["status"], "sent")
        later = tools.send_interaction("room", "Which now?", decision=True, review_complete=True)
        self.assertEqual(later["status"], "review_required")

    def test_decision_review_off_sends_at_once_without_a_request(self) -> None:
        tools, calls = self.reviewed(decision_review=False)
        result = tools.send_interaction("room", "Which toolchain?", decision=True)
        self.assertEqual(result["status"], "sent")
        self.assertNotIn("review", result)
        self.assertNotIn("without one", result["summary"])
        self.assertEqual(calls, [])
        self.assertEqual([r["reason"] for r in self.lifetime("decision_review_skipped")], ["disabled"])

    def test_a_registry_without_a_reviewer_sends_at_once(self) -> None:
        self.client.send("room", sender="entity_1", content="Please build it.")
        result = self.tools().send_interaction("room", "Which?", decision=True)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["review"]["reason"], "unavailable")

    def test_the_review_is_recorded_in_the_lifetime_log_and_the_trace(self) -> None:
        tools, _ = self.reviewed()
        tools.send_interaction("room", "Which toolchain?", decision=True,
                               options=["clang", "GCC"])
        record = self.lifetime("decision_review_completed")[0]
        self.assertEqual((record["request_id"], record["connection"], record["necessary"],
                          record["framing"], record["default"], record["interaction_id"]),
                         ("request_1", "oversight", "yes", "revise", "GCC", "room"))
        self.assertEqual(record["points"], ["name the options", "state your recommendation"])
        trace = [json.loads(line) for line in self.paths.life_loop_log.read_text().splitlines()]
        self.assertIn("decision_review_completed", [item["kind"] for item in trace])

    def test_the_verdict_is_parsed_from_the_fixed_form(self) -> None:
        from artificium.tool_review import parse_decision_review as parse
        text = ("<think>hm</think>**NECESSARY:** No\n**FRAMING**: OK\nPOINTS: none\n"
                "DEFAULT: `Clang.`\nREASONS: the entity said\n  go ahead.")
        self.assertEqual(parse(text, ["gcc", "clang"]), {
            "necessary": "no", "framing": "ok", "points": [], "default": "clang",
            "reasons": "the entity said\n  go ahead."})
        for words in ("none", "None - no option is safe", "no safe default", "n/a"):
            self.assertIsNone(parse(f"NECESSARY: yes\nFRAMING: ok\nDEFAULT: {words}")["default"])
        # An option that merely begins with "none" is still an option.
        self.assertEqual(parse("NECESSARY: yes\nFRAMING: ok\nDEFAULT: None of the above",
                               ["None of the above", "x"])["default"], "None of the above")
        # A free answer keeps what the reviewer wrote.
        self.assertEqual(parse("NECESSARY: yes\nFRAMING: ok\nDEFAULT: proceed")["default"], "proceed")
        self.assertIsNone(parse("NECESSARY: yes"))
        self.assertIsNone(parse(""))

    def test_the_prompts_are_declared_and_fully_filled(self) -> None:
        request = self.prompts.runtime(
            "decision_review_request", ask="A", open_decision="O", message="M",
            options="Opts", default="D", reply_by="2 h")
        self.assertNotIn("{{", request)
        system = self.prompts.runtime("decision_review_system")
        for label in ("NECESSARY:", "FRAMING:", "POINTS:", "DEFAULT:", "REASONS:"):
            self.assertIn(label, system)

    def test_the_tool_catalog_and_native_schema_know_review_complete(self) -> None:
        from artificium.tool_schemas import native_tool_schemas
        tools = self.tools()
        self.assertIn("review_complete", tools.accepted_arguments("send_interaction")["optional"])
        self.assertIn("review_complete", tools.canonical_example("send_interaction"))
        schemas = {item["function"]["name"]: item["function"]
                   for item in native_tool_schemas(ToolRegistry, self.prompts.tool_catalog())}
        parameters = schemas["send_interaction"]["parameters"]
        self.assertEqual(parameters["properties"]["review_complete"], {"type": "boolean"})
        self.assertNotIn("review_complete", parameters["required"])
        self.assertIn("review_complete", schemas["send_interaction"]["description"])

    def test_the_setting_defaults_on_validates_and_is_saved_with_every_field(self) -> None:
        self.assertIs(self.config.decision_review, True)
        with self.assertRaises(ValueError):
            dataclasses.replace(self.config, decision_review="no")
        document = json.loads(self.paths.config.read_text())
        self.assertIs(document["harness"]["decision_review"], True)
        del document["harness"]["decision_review"]
        self.paths.config.write_text(json.dumps(document))
        self.assertIs(ConfigStore(self.paths).load().decision_review, True)

    def test_configure_turns_it_off_and_on_offline(self) -> None:
        import contextlib
        import io
        from artificium.cli import main

        def run(*argv):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                return main(["--root", str(self.root), *argv]), err.getvalue()

        code, err = run("configure", "harness", "--decision-review", "off")
        self.assertEqual(code, 0, err)
        self.assertIs(ConfigStore(self.paths).load().decision_review, False)
        self.assertEqual(run("configure", "harness", "--decision-review", "on")[0], 0)
        self.assertIs(ConfigStore(self.paths).load().decision_review, True)
        # It belongs to the harness, not to a connection.
        self.assertEqual(run("configure", "oversight", "--decision-review", "off")[0], 2)
        self.assertEqual(run("configure", "model", "--decision-review", "off")[0], 2)

    def test_watch_shows_a_short_line_for_each_record(self) -> None:
        from artificium import display
        options = display.RenderOptions(color=False, width=100, timestamps=False)
        done = display.render_event_body({
            "kind": "decision_review_completed", "necessary": "no", "framing": "ok",
            "default": None, "preview": "The entity already said so."}, options)
        self.assertEqual(done, ["[decision review] necessary: no, framing: ok, default: none "
                                "— The entity already said so."])
        skipped = display.render_event_body(
            {"kind": "decision_review_skipped", "reason": "cost_limit"}, options)
        self.assertEqual(skipped, ["[decision review] did not run (cost_limit); the decision "
                                   "was sent without it"])


class IdleFallbackTests(OversightCase):
    def agent(self, **changes) -> Artificium:
        if changes:
            ConfigStore(self.paths).save(dataclasses.replace(self.config, **changes))
        return Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))

    def test_only_a_sleep_with_no_wake_gets_a_fallback(self) -> None:
        agent = self.agent()
        task = [{"name": "x", "run_at": "2030-01-01T00:00:00Z", "description": ""}]
        self.assertIsNotNone(agent._idle_fallback_at("until_event", []))
        self.assertIsNone(agent._idle_fallback_at("until_event", task))
        self.assertIsNone(agent._idle_fallback_at("timed", []))
        self.assertIsNone(self.agent(idle_wake_hours=0)._idle_fallback_at("until_event", []))
        self.ask(agent.tools, default="proceed")
        self.assertIsNone(agent._idle_fallback_at("until_event", []))

    def test_the_fallback_ends_the_sleep_with_a_reason_the_agent_can_act_on(self) -> None:
        agent = self.agent(idle_wake_hours=1)
        atomic_write_json(self.paths.sleep_state, {
            "active": True, "mode": "until_event", "started_at": "2026-01-01T00:00:00Z",
            "idle_fallback_at_epoch": 1.0})
        self.assertFalse(agent._sleep_active())
        self.assertEqual(read_json(self.paths.sleep_state)["reason"], "idle_fallback")
        claimed = agent._claim_notifications()
        rendered = agent._render_notifications(claimed)
        self.assertTrue(any("IDLE WAKE" in text and "after 1 hours" in text for text in rendered))

        atomic_write_json(self.paths.sleep_state, {
            "active": True, "mode": "until_event", "started_at": "2026-01-01T00:00:00Z",
            "idle_fallback_at_epoch": _later(5).timestamp()})
        self.assertTrue(agent._sleep_active())


class UncleanExitTests(OversightCase):
    def test_a_state_left_running_is_reported_once_the_lock_is_ours(self) -> None:
        agent = Artificium(self.paths, engine=FakeEngine([]), console=Console(quiet=True))
        agent._note_previous_exit()
        self.assertNotIn("previous_run_ended_uncleanly", self.paths.lifetime_log.read_text())
        for status in ("stopped",):
            atomic_write_json(self.paths.runtime_state, {"status": status, "pid": 1})
            agent._note_previous_exit()
        self.assertNotIn("previous_run_ended_uncleanly", self.paths.lifetime_log.read_text())
        atomic_write_json(self.paths.runtime_state, {
            "status": "running", "pid": 4242, "updated_at": "2026-01-01T00:00:00Z"})
        with open(self.paths.lifetime_log, "a") as handle:
            handle.write(json.dumps({"timestamp": "2026-01-01T00:00:05Z", "kind": "x",
                                     "run_id": "run_an_earlier_process"}) + "\n")
        agent.records.emit("this_process_started")  # logged after the old one died
        agent._note_previous_exit()
        records = [json.loads(line) for line in self.paths.lifetime_log.read_text().splitlines()
                   if "previous_run_ended_uncleanly" in line]
        self.assertEqual(records[0]["previous_pid"], 4242)
        self.assertEqual(records[0]["previous_status"], "running")
        self.assertEqual(records[0]["last_record_at"], "2026-01-01T00:00:05Z")


if __name__ == "__main__":
    unittest.main()
