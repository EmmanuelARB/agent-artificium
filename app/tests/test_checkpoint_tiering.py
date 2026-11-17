from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from artificium.checkpoints import BACKLOG_FACTOR, MERGE_BATCH, CheckpointTiering
from artificium.cli import build_parser, main
from artificium.config import Config, ConfigStore
from artificium.engine import Engine, EngineReply
from artificium.filesystem import Paths, read_json
from artificium.prompts import PromptPack
from artificium.records import Console, Records
from artificium.runtime import Artificium

ROOT = Path(__file__).resolve().parents[2]
# More than the migration threshold (BACKLOG_FACTOR times the verbatim limit).
BACKLOG_SIZE = 3 * BACKLOG_FACTOR + 5
SUMMARY = (
    "Objective: finish the comparison. Facts: the first method was refuted by the "
    "second measurement. Next step: rerun the comparison with the corrected input."
)


class ScriptedEngine(Engine):
    def complete(self, messages):
        return EngineReply("No action.")


class FakeSummarizer:
    """The merge's model call, without a model."""

    def __init__(self, text: str = SUMMARY, fail: Exception | None = None):
        self.text = text
        self.fail = fail
        self.calls: list[list[dict]] = []

    def __call__(self, sources):
        self.calls.append(sources)
        if self.fail is not None:
            raise self.fail
        return self.text


class TieringCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        self.paths.ensure_layout()
        self.config = Config(provider="custom", model="test", base_url="http://example.invalid/v1",
                      checkpoint_keep_verbatim=3)
        self.records = Records(self.paths)
        self.summarizer = FakeSummarizer()
        self.tiering = CheckpointTiering(
            self.paths, self.config, self.records,
            console=Console(quiet=True), summarize=self.summarizer,
        )

    def checkpoint(self, name: str, text: str, *, created_at: str, tokens: int = 100) -> Path:
        path = self.paths.checkpoint_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.assertTrue(self.tiering.register(path, tokens=tokens, created_at=created_at))
        return path

    def stamp(self, index: int) -> str:
        return f"2026-01-{index + 1:02d}T00:00:00.000000Z"

    def fill(self, count: int, *, prefix: str = "checkpoint") -> list[Path]:
        return [
            self.checkpoint(
                f"{prefix}-{index}.txt",
                f"Checkpoint {index}: objective state, evidence path, and the next step.",
                created_at=self.stamp(index),
                tokens=100 + index,
            )
            for index in range(count)
        ]

    def existing(self) -> list[str]:
        return sorted(
            str(path.relative_to(self.paths.checkpoint_dir))
            for path in self.paths.checkpoint_dir.rglob("*")
            if path.is_file()
        )


class MergeTests(TieringCase):
    def test_merge_triggers_only_above_the_keep_threshold(self):
        self.fill(3)
        report = self.tiering.tier()
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(report["merges"], 0)
        self.assertEqual(self.summarizer.calls, [])
        self.assertEqual(len(self.existing()), 3)

        self.checkpoint("checkpoint-3.txt", "Checkpoint 3: next step recorded.",
                        created_at=self.stamp(3))
        report = self.tiering.tier()
        self.assertEqual(report["status"], "merged")
        self.assertEqual(report["merges"], 1)
        self.assertEqual(len(self.summarizer.calls), 1)
        self.assertEqual(report["verbatim"], 3)
        self.assertEqual(report["merged"], 1)

    def test_newest_checkpoints_stay_verbatim(self):
        paths = self.fill(6)
        report = self.tiering.tier()
        self.assertEqual(report["merges"], 1)
        for index in (3, 4, 5):
            self.assertTrue(paths[index].is_file(), paths[index])
            self.assertIn(f"Checkpoint {index}", paths[index].read_text(encoding="utf-8"))
        for index in (0, 1, 2):
            self.assertFalse(paths[index].exists(), paths[index])
        self.assertEqual(
            sorted(entry["relative_path"].rsplit("/", 1)[-1]
                   for entry in self.tiering.entries() if entry["tier"] == "verbatim"),
            ["checkpoint-3.txt", "checkpoint-4.txt", "checkpoint-5.txt"],
        )

    def test_merged_file_names_its_sources_and_their_paths(self):
        self.fill(5)
        report = self.tiering.tier()
        target = self.paths.root / report["target"]
        content = target.read_text(encoding="utf-8")
        self.assertTrue(target.is_file())
        self.assertIn("merged from 2 older checkpoints", content)
        for index in (0, 1):
            self.assertIn(f"checkpoint-{index}.txt", content)
            self.assertIn(f"mind/memory/context/checkpoint-{index}.txt", content)
        self.assertIn("Next step: rerun the comparison", content)
        self.assertEqual(self.summarizer.calls[0][0]["path"],
                         "mind/memory/context/checkpoint-0.txt")

    def test_the_merged_tier_stays_one_file(self):
        self.fill(5)
        previous = self.tiering.tier()["target"]
        self.fill(4, prefix="later")
        report = self.tiering.tier()
        self.assertEqual(report["merged"], 1)
        entries = [entry for entry in self.tiering.entries() if entry["tier"] == "merged"]
        self.assertEqual(len(entries), 1)
        # The previous merged file is folded into the new one and retired.
        self.assertIn(previous, report["sources"])
        self.assertFalse((self.paths.root / previous).exists())

    def test_archives_are_untouched(self):
        archive = self.paths.context_archive / "offload--1.jsonl"
        archive.write_text('{"role":"user","content":"raw"}\n', encoding="utf-8")
        before = archive.read_bytes()
        self.fill(6)
        self.tiering.tier()
        self.assertEqual(archive.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.paths.context_archive.iterdir()),
                         ["offload--1.jsonl"])

    def test_a_failed_merge_leaves_every_checkpoint_intact(self):
        paths = self.fill(6)
        index_before = self.paths.checkpoint_index.read_bytes()
        self.summarizer.fail = RuntimeError("connection refused")
        report = self.tiering.tier()
        self.assertEqual(report["status"], "failed")
        self.assertIn("connection refused", report["errors"][0])
        self.assertEqual(report["merges"], 0)
        for path in paths:
            self.assertTrue(path.is_file(), path)
        self.assertEqual(self.paths.checkpoint_index.read_bytes(), index_before)

    def test_a_too_short_summary_is_refused(self):
        paths = self.fill(6)
        self.summarizer.text = "too short"
        report = self.tiering.tier()
        self.assertEqual(report["status"], "failed")
        self.assertTrue(all(path.is_file() for path in paths))
        self.assertEqual(len(self.existing()), 6)

    def test_a_large_excess_merges_in_batches(self):
        self.fill(MERGE_BATCH + 2)
        report = self.tiering.tier()
        self.assertEqual(report["verbatim"], 3)
        self.assertEqual(report["merged"], 1)
        self.assertEqual(len(self.summarizer.calls), 1)
        self.assertEqual(len(self.summarizer.calls[0]), MERGE_BATCH - 1)
        # A backlog larger than one batch takes more than one merge.
        self.fill(MERGE_BATCH + 5, prefix="more")
        self.assertEqual(self.tiering.tier()["merges"], 2)

    def test_tiering_off_keeps_every_checkpoint(self):
        self.config = Config(provider="custom", model="test", base_url="http://example.invalid/v1",
                         checkpoint_keep_verbatim=0)
        self.tiering = CheckpointTiering(self.paths, self.config, self.records,
                                         summarize=self.summarizer)
        self.fill(6)
        report = self.tiering.tier()
        self.assertEqual(report["status"], "disabled")
        self.assertEqual(len(self.existing()), 6)

    def test_a_limited_run_stops_and_the_next_one_continues(self):
        self.fill(4 * MERGE_BATCH)
        first = self.tiering.tier(limit=1)
        self.assertEqual(first["merges"], 1)
        second = self.tiering.tier()
        self.assertEqual(second["merges"], 3)
        self.assertEqual(second["verbatim"], 3)


class BacklogTests(TieringCase):
    def backlog_files(self, count: int) -> list[Path]:
        paths = []
        for index in range(count):
            path = self.paths.checkpoint_dir / f"old-{index}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"Old checkpoint {index}: what was established.\n",
                            encoding="utf-8")
            paths.append(path)
        return paths

    def test_a_fresh_instance_gets_no_marker(self):
        self.fill(2)
        report = self.tiering.backlog()
        self.assertEqual(report["status"], "not_needed")
        self.assertFalse(self.paths.checkpoint_tiering_state.exists())

    def test_the_backlog_runs_once_and_is_resumable(self):
        self.backlog_files(BACKLOG_SIZE)
        first = self.tiering.backlog(limit=1)
        self.assertEqual(first["backlog"], True)
        self.assertEqual(first["merges"], 1)
        self.assertEqual(first["status"], "merged")
        state = read_json(self.paths.checkpoint_tiering_state)
        self.assertTrue(state["started_at"])
        self.assertIsNone(state["completed_at"])
        self.assertEqual(state["merges"], 1)
        # Resuming continues the same migration rather than starting over.
        second = self.tiering.backlog()
        self.assertEqual(second["verbatim"], 3)
        state = read_json(self.paths.checkpoint_tiering_state)
        self.assertTrue(state["completed_at"])
        self.assertEqual(state["merges"], 1 + second["merges"])
        # A completed marker means never again, even when files reappear.
        self.backlog_files(BACKLOG_SIZE)
        third = self.tiering.backlog()
        self.assertEqual(third["status"], "already_completed")
        self.assertEqual(third["merges"], 0)

    def test_the_backlog_never_loses_a_checkpoint(self):
        paths = self.backlog_files(BACKLOG_SIZE)
        retired = 0
        while True:
            before = {path.name for path in self.paths.checkpoint_dir.iterdir() if path.is_file()}
            report = self.tiering.tier(limit=1, adopt=True)
            if not report["merges"]:
                break
            after = {path.name for path in self.paths.checkpoint_dir.iterdir() if path.is_file()}
            content = (self.paths.root / report["target"]).read_text(encoding="utf-8")
            for removed in before - after:
                # A source is deleted only after the target names it.
                self.assertIn(removed, content)
                retired += 1
        # Every source, plus each intermediate merged file it superseded.
        self.assertGreaterEqual(retired, BACKLOG_SIZE - 3)
        self.assertEqual(len([path for path in paths if path.is_file()]), 3)

    def test_an_offline_start_skips_and_retries_later(self):
        self.backlog_files(BACKLOG_SIZE)
        self.summarizer.fail = RuntimeError("no connection")
        report = self.tiering.backlog()
        self.assertEqual(report["status"], "failed")
        state = read_json(self.paths.checkpoint_tiering_state)
        self.assertIsNone(state["completed_at"])
        self.assertIn("no connection", state["last_error"])
        self.assertEqual(len(self.tiering.entries()) + len(self.tiering.adopt(write=False)),
                         BACKLOG_SIZE)
        self.summarizer.fail = None
        resumed = self.tiering.backlog()
        self.assertEqual(resumed["verbatim"], 3)
        self.assertTrue(read_json(self.paths.checkpoint_tiering_state)["completed_at"])

    def test_a_stopped_pass_leaves_the_marker_open(self):
        self.backlog_files(BACKLOG_SIZE)
        checks = {"count": 0}

        def stop():
            checks["count"] += 1
            return checks["count"] > 1

        report = self.tiering.backlog(stop=stop)
        self.assertEqual(report["merges"], 1)
        state = read_json(self.paths.checkpoint_tiering_state)
        self.assertIsNone(state["completed_at"])
        self.assertGreater(len(self.tiering.entries()), 3)

    def test_adoption_uses_modification_time_not_the_name(self):
        older = self.paths.checkpoint_dir / "zzz-newest-name.txt"
        newer = self.paths.checkpoint_dir / "aaa-oldest-name.txt"
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for path, stamp in ((older, 1_000_000), (newer, 2_000_000)):
            path.write_text(f"{path.name}: content that is long enough to be a checkpoint.\n",
                            encoding="utf-8")
            os.utime(path, (stamp, stamp))
        self.tiering.adopt()
        entries = self.tiering.entries()
        self.assertEqual([entry["path"].name for entry in entries],
                         ["zzz-newest-name.txt", "aaa-oldest-name.txt"])


class DryRunTests(TieringCase):
    def test_dry_run_changes_nothing(self):
        paths = self.fill(6)
        listing = self.existing()
        index = self.paths.checkpoint_index.read_bytes()
        report = self.tiering.tier(dry_run=True)
        self.assertEqual(report["status"], "planned")
        self.assertEqual(report["merges"], 1)
        self.assertEqual(report["verbatim"], 3)
        self.assertEqual(self.summarizer.calls, [])
        self.assertEqual(self.existing(), listing)
        self.assertEqual(self.paths.checkpoint_index.read_bytes(), index)
        self.assertTrue(all(path.is_file() for path in paths))

    def test_a_dry_run_backlog_adopts_nothing(self):
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        files = [self.paths.checkpoint_dir / f"old-{index}.txt"
                 for index in range(BACKLOG_SIZE)]
        for path in files:
            path.write_text("Old checkpoint content that is long enough.\n", encoding="utf-8")
        report = self.tiering.backlog(dry_run=True)
        self.assertEqual(report["merges"], 2)
        self.assertEqual(report["verbatim"], 3)
        self.assertEqual(len(self.tiering.entries()), 0)
        self.assertFalse(self.paths.checkpoint_index.exists())
        self.assertFalse(self.paths.checkpoint_tiering_state.exists())
        self.assertTrue(all(path.is_file() for path in files))


class ConfigTests(unittest.TestCase):
    def test_round_trip(self):
        config = Config(provider="custom", model="test", base_url="http://example.invalid/v1",
                      checkpoint_keep_verbatim=7)
        restored = Config.from_dict(config.grouped_dict())
        self.assertEqual(restored.checkpoint_keep_verbatim, 7)
        self.assertEqual(config.grouped_dict()["harness"]["checkpoint_keep_verbatim"], 7)

    def test_a_missing_key_is_the_current_default(self):
        grouped = Config(provider="custom", model="test", base_url="http://example.invalid/v1").grouped_dict()
        del grouped["harness"]["checkpoint_keep_verbatim"]
        self.assertEqual(Config.from_dict(grouped).checkpoint_keep_verbatim, 3)

    def test_invalid_values_are_refused(self):
        for value in (-1, 51, True, "3", 1.5, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    Config(provider="custom", model="test", base_url="http://example.invalid/v1",
                       checkpoint_keep_verbatim=value)
        self.assertEqual(
            Config(provider="custom", model="test", base_url="http://example.invalid/v1",
                         checkpoint_keep_verbatim=0)
            .checkpoint_keep_verbatim,
            0,
        )

    def test_the_store_writes_every_field(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        paths = Paths(Path(temporary.name))
        paths.ensure_layout()
        ConfigStore(paths).save(Config(provider="custom", model="test",
                                       base_url="http://example.invalid/v1"))
        saved = json.loads(paths.config.read_text(encoding="utf-8"))
        self.assertEqual(saved["harness"]["checkpoint_keep_verbatim"], 3)


class RuntimeTests(unittest.TestCase):
    """The offload path registers its checkpoint and tiers older ones."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.config = Config(
            provider="llamacpp", model="local-model",
            base_url="http://127.0.0.1:8080", context_window_tokens=50_000,
            request_timeout_seconds=5.0, checkpoint_keep_verbatim=2,
        )
        ConfigStore(self.paths).save(self.config)
        self.summarizer = FakeSummarizer()
        self.agent = Artificium(self.paths, engine=ScriptedEngine(),
                                console=Console(quiet=True))
        self.agent.initialization.finish(
            "Self, meta-memory, tools, environment, and pending events inspected.")
        self.agent.checkpoints.summarize = self.summarizer
        # Three older checkpoints, as an instance that already offloaded has.
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            path = self.paths.checkpoint_dir / f"earlier-{index}.txt"
            path.write_text(f"Earlier checkpoint {index}: objective state and next step.\n",
                            encoding="utf-8")
            self.agent.checkpoints.register(path, tokens=100,
                                            created_at=f"2026-01-0{index + 1}T00:00:00Z")

    def offload(self, path: str) -> dict:
        return self.agent.tools.offload_working_memory(
            path=path,
            checkpoint=("Objective: finish the comparison. Facts: the first method was "
                        "refuted. Next step: rerun it with the corrected input."),
            retrieve_when="When the comparison resumes.",
            reflection_complete=True,
        )

    def test_an_offload_registers_its_checkpoint_and_tiers_the_rest(self):
        # The pending reflection a real turn would have requested first.
        self.agent.tools.offload_working_memory(path="context/current-comparison")
        result = self.offload("context/current-comparison")
        self.assertEqual(result["status"], "offloaded")
        self.agent._record_checkpoint(result)
        entries = self.agent.checkpoints.entries()
        self.assertEqual(len([entry for entry in entries if entry["tier"] == "verbatim"]), 2)
        self.assertEqual(len([entry for entry in entries if entry["tier"] == "merged"]), 1)
        self.assertEqual(len(self.summarizer.calls), 1)
        # The agent is told which paths its indexes can no longer resolve.
        context = self.agent.working.load()
        self.assertIn("OLD CHECKPOINTS MERGED", context[-1]["content"])
        self.assertIn("earlier-0.txt", context[-1]["content"])
        kinds = [json.loads(line)["kind"]
                 for line in self.paths.life_loop_log.read_text(encoding="utf-8").splitlines()]
        self.assertIn("checkpoint_merged", kinds)

    def test_a_checkpoint_outside_the_checkpoint_directory_is_left_alone(self):
        self.agent.tools.offload_working_memory(path="projects/current-comparison")
        result = self.offload("projects/current-comparison")
        self.agent._record_checkpoint(result)
        self.assertEqual(len(self.agent.checkpoints.entries()), 3)
        self.assertEqual(self.summarizer.calls, [])
        self.assertTrue((self.paths.mind / result["memory_path"]).is_file())

    def test_the_backlog_pass_never_raises_into_the_life_loop(self):
        self.agent.checkpoints.summarize = FakeSummarizer(fail=RuntimeError("offline"))
        for index in range(3 * BACKLOG_FACTOR + 5):
            path = self.paths.checkpoint_dir / f"backlog-{index}.txt"
            path.write_text("A backlog checkpoint with enough content to summarize.\n",
                            encoding="utf-8")
        report = self.agent.tier_checkpoint_backlog()
        self.assertEqual(report["status"], "failed")
        kinds = [json.loads(line)["kind"]
                 for line in self.paths.lifetime_log.read_text(encoding="utf-8").splitlines()]
        self.assertIn("checkpoint_tiering_skipped", kinds)


class CliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        shutil.copytree(ROOT / "app/seed", self.paths.seed)
        self.paths.ensure_layout()
        self.config = Config(provider="llamacpp", model="local-model",
                             base_url="http://127.0.0.1:8080", checkpoint_keep_verbatim=2)
        ConfigStore(self.paths).save(self.config)

    def test_the_parser_accepts_the_command(self):
        args = build_parser().parse_args(["checkpoints", "tier", "--dry-run", "--limit", "2"])
        self.assertEqual((args.command, args.action, args.dry_run, args.limit),
                         ("checkpoints", "tier", True, 2))
        self.assertEqual(build_parser().parse_args(["checkpoints"]).action, "tier")

    def test_a_dry_run_reports_without_changing_anything(self):
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for index in range(4):
            (self.paths.checkpoint_dir / f"old-{index}.txt").write_text(
                "An unregistered checkpoint with enough content to summarize.\n",
                encoding="utf-8")
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--root", str(self.paths.install), "checkpoints", "tier", "--dry-run"])
        self.assertEqual(code, 0)
        self.assertIn("planned", output.getvalue())
        self.assertEqual(len(list(self.paths.checkpoint_dir.iterdir())), 4)
        self.assertFalse(self.paths.checkpoint_index.exists())


class PromptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT / "app/prompts", self.paths.prompts)
        self.pack = PromptPack(self.paths)

    def test_the_reflection_prompt_teaches_progress_checkpoints(self):
        text = self.pack.event("working_memory_offload_reflection")
        guidance = (
            "During long multi-hour tasks, save a short progress checkpoint with "
            "`save_memory` every so often: facts established, hypotheses refuted, "
            "next step, measure. Those memories outlive this context and inform "
            "later summaries."
        )
        self.assertIn(guidance, " ".join(text.split()))
        # The delta is deliberately small: this event prompt is paid at every
        # offload, so growth is measured rather than assumed.
        self.assertLessEqual(len(guidance.split()), 40)
        self.assertLessEqual(len(text.split()), 700)

    def test_the_merge_prompt_is_short_and_has_no_placeholders(self):
        text = self.pack.runtime("checkpoint_tier_merge")
        self.assertLessEqual(len(text.split()), 160)
        self.assertNotIn("{{", text)

    def test_the_merged_notice_names_the_target_and_the_sources(self):
        text = self.pack.event(
            "checkpoint_merged", count=2,
            target="mind/memory/context/older-tier--1.txt",
            sources="- mind/memory/context/a.txt\n- mind/memory/context/b.txt",
        )
        self.assertIn("mind/memory/context/older-tier--1.txt", text)
        self.assertIn("mind/memory/context/a.txt", text)
        self.assertIn("logs/context/", text)


if __name__ == "__main__":
    unittest.main()
