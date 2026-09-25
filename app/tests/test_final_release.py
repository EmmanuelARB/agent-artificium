from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from artificium.cli import main
from artificium.config import Config, ConfigStore
from artificium.engine import Engine, EngineError, EngineReply
from artificium.filesystem import Paths, atomic_write_json, read_json
from artificium.interactions import ArtificiumClient
from artificium.records import Console
from artificium.runtime import Artificium
from artificium.setup import ModelDiscovery, SetupOptions, SetupWizard

ROOT = Path(__file__).resolve().parents[2]


def call(name, **arguments):
    return '<tool_call>' + json.dumps({'tool': name, **arguments}) + '</tool_call>'


class ScriptedEngine(Engine):
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages):
        self.requests.append(messages)
        response = self.responses.pop(0) if self.responses else '<think>Waiting.</think>'
        if isinstance(response, Exception):
            raise response
        return EngineReply(response)


class FinalReleaseCase(unittest.TestCase):
    def setUp(self):
        # These unit cases isolate metadata/core behavior. Real connection
        # verification and failures are covered over HTTP in test_connection_flow.
        check = mock.patch('artificium.setup.verify_connection', side_effect=lambda paths, config, key, **kw: (config, {}))
        check.start()
        self.addCleanup(check.stop)
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.paths = Paths(Path(temporary.name))
        shutil.copytree(ROOT/'app/prompts', self.paths.prompts)
        shutil.copytree(ROOT/'app/seed', self.paths.seed)
        self.paths.ensure_layout()
        self.config = Config(provider='custom', model='test', base_url='http://example.invalid/v1',
                             context_window_tokens=50000, max_life_loop_rounds=1)
        ConfigStore(self.paths).save(self.config)
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        network = mock.patch('urllib.request.urlopen', side_effect=OSError('offline fixture'))
        network.start()
        self.addCleanup(network.stop)

    def agent(self, engine=None, **settings):
        self.config = dataclasses.replace(self.config, **settings)
        ConfigStore(self.paths).save(self.config)
        agent = Artificium(self.paths, engine=engine or ScriptedEngine(), console=Console(quiet=True))
        agent.initialization.finish('Self, meta-memory, tools, environment, and pending events inspected.')
        return agent

    def fill_context(self, agent):
        # Pressure remains below the serving limit so ordinary/offload actions
        # have room to run. Actual overflow is covered by context recovery tests.
        agent.working.append({'role':'assistant','content':'source detail ' * 8000}, origin='test')

    def image(self, name='image.png'):
        image = self.paths.root/name
        image.write_bytes(b'offline image fixture')
        return image

    def test_image_retry_keeps_exact_current_notification_batch_once(self):
        engine = ScriptedEngine(EngineError('image input is unsupported', status=415), '<think>Recovered.</think>')
        agent = self.agent(engine)
        agent._append_runtime(['A prior notification already stored.'], origin='test')
        event, _ = ArtificiumClient(self.paths.install).send('room', sender='user', content='New event remains durable.')
        agent.visual.load([str(self.image())], retention='persistent')
        agent.run_turn()
        batches = [[m['content'] for m in request if m.get('_artificium',{}).get('kind')=='runtime_input'] for request in engine.requests]
        self.assertEqual(len(engine.requests), 2)
        for content in batches[0]:
            self.assertEqual(batches[1].count(content), 1)
        self.assertEqual(len(batches[1]), len(batches[0]) + 1)  # fallback explanation
        self.assertNotIn('artificium_image', json.dumps(engine.requests[1]))
        stored = [m['content'] for m in agent.working.load()]
        self.assertEqual(stored.count(batches[0][-1]), 1)
        receipt = read_json(self.paths.receipts/(event['id']+'.json'))
        self.assertTrue(receipt['delivered_at'])
        self.assertFalse(receipt['handled_at'])

    def test_failed_image_retry_returns_notifications_to_queue(self):
        engine = ScriptedEngine(EngineError('image input is unsupported',status=415), EngineError('server unavailable',status=503))
        agent = self.agent(engine)
        notification = agent.notifications.create(type='external_event',summary='Keep this notification',source='test')
        agent.visual.load([str(self.image())])
        with self.assertRaises(EngineError):
            agent.run_turn()
        self.assertEqual(len(engine.requests), 2)
        self.assertTrue((self.paths.notifications_new/(notification.id+'.json')).exists())
        self.assertFalse(list(self.paths.notifications_delivered.glob('*.json')))

    def test_explicit_vision_yes_keeps_existing_no_fallback_behavior(self):
        engine = ScriptedEngine(EngineError('image input is unsupported',status=415))
        agent = self.agent(engine, vision='yes', vision_preference='yes')
        agent.visual.load([str(self.image())])
        with self.assertRaises(EngineError):
            agent.run_turn()
        self.assertEqual(len(engine.requests), 1)

    def test_disabling_vision_suspends_saved_images_without_consuming_them(self):
        agent = self.agent()
        first, second = self.image('once.png'), self.image('persistent.png')
        agent.visual.load([str(first)], retention='once')
        agent.visual.load([str(second)], retention='persistent')
        before = self.paths.visual_context.read_bytes()
        engine = ScriptedEngine('<think>Text only.</think>')
        restarted = self.agent(engine, vision='no', vision_preference='no')
        self.assertIsNone(restarted.visual.request_message())
        self.assertEqual(restarted.visual.load([str(first)])['status'], 'vision_unavailable')
        restarted.run_turn()
        self.assertNotIn('artificium_image', json.dumps(engine.requests))
        self.assertEqual(self.paths.visual_context.read_bytes(), before)
        restored = self.agent(vision='auto', vision_preference='auto', model_supports_vision=True)
        self.assertEqual(len(restored.visual.request_message()['_artificium']['image_ids']), 2)
        self.assertTrue(first.exists() and second.exists())

    def test_legacy_config_is_read_without_rewrite_and_migrates_on_save(self):
        legacy = dataclasses.asdict(self.config)
        for name in ('vision_preference','model_supports_vision','mandatory_offload','offload_threshold_percent'):
            legacy.pop(name)
        legacy.update(schema_version=2, vision='no', heartbeat_seconds=None, top_k=17)
        atomic_write_json(self.paths.config, legacy)
        before = self.paths.config.read_bytes()
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(self.paths.config.read_bytes(), before)
        self.assertEqual(loaded.vision_preference, 'no')
        self.assertFalse(loaded.mandatory_offload)
        ConfigStore(self.paths).save(loaded)
        grouped = read_json(self.paths.config)
        self.assertEqual(set(grouped), {'schema_version','harness','model'})
        self.assertNotIn('heartbeat_seconds', grouped['harness'])
        self.assertEqual(grouped['model']['top_k'], 17)
        self.assertEqual(ConfigStore(self.paths).load(), loaded)

    def test_grouped_config_ignores_retired_heartbeat_values(self):
        for value in (None, 30, 3600):
            with self.subTest(heartbeat=value):
                grouped = self.config.grouped_dict()
                grouped['harness']['heartbeat_seconds'] = value
                atomic_write_json(self.paths.config, grouped)
                before = self.paths.config.read_bytes()
                loaded = ConfigStore(self.paths).load()
                self.assertEqual(loaded, self.config)
                self.assertEqual(self.paths.config.read_bytes(), before)
                ConfigStore(self.paths).save(loaded)
                self.assertNotIn('heartbeat_seconds', read_json(self.paths.config)['harness'])

    def test_harness_command_works_offline_without_changing_connection(self):
        before = ConfigStore(self.paths).load().grouped_dict()['model']
        with mock.patch('artificium.setup.discover_provider_models',side_effect=AssertionError('must not probe')), redirect_stdout(io.StringIO()):
            result = main(['--root',str(self.paths.install),'configure','harness',
                           '--mandatory-offload','on','--offload-threshold','75'])
        self.assertEqual(result, 0)
        loaded = ConfigStore(self.paths).load()
        self.assertEqual(loaded.grouped_dict()['model'], before)
        self.assertEqual((loaded.mandatory_offload,loaded.offload_threshold_percent),(True,75))

    def test_switching_models_preserves_harness_preferences_and_refreshes_vision(self):
        initial = self.agent(vision='no',vision_preference='auto',model_supports_vision=False,
                             mandatory_offload=True,offload_threshold_percent=72).config
        discovery = ModelDiscovery(models=('new',),details={'new':{'context_length':64000,'vision':True}})
        with mock.patch('artificium.setup.discover_provider_models',return_value=discovery):
            updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope='model',model='new'))
        self.assertEqual(updated.grouped_dict()['harness'],initial.grouped_dict()['harness'])
        self.assertEqual((updated.vision_preference,updated.vision),('auto','auto'))

    def test_known_text_only_model_respects_preference_without_sending_images(self):
        self.agent(vision='no',vision_preference='auto',model_supports_vision=False)
        updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope='harness',vision='yes'))
        self.assertEqual((updated.vision_preference,updated.vision),('yes','no'))

    def test_scope_mistakes_and_invalid_threshold_never_save(self):
        before = self.paths.config.read_bytes()
        for options in (SetupOptions(scope='model',vision='no'), SetupOptions(scope='harness',model='other'),
                        SetupOptions(scope='harness',offload_threshold_percent=100),
                        SetupOptions(scope='harness',offload_threshold_percent=float('nan'))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SetupWizard(self.paths).reconfigure(options)
            self.assertEqual(self.paths.config.read_bytes(),before)

    def test_legacy_combined_cli_flags_and_grouped_config_display(self):
        with redirect_stdout(io.StringIO()):
            result = main(['--root',str(self.paths.install),'configure','--mandatory-offload','on'])
        self.assertEqual(result,0)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['--root',str(self.paths.install),'config','harness']),0)
        self.assertTrue(json.loads(output.getvalue())['mandatory_offload'])

    def test_interactive_harness_edit_needs_no_model_or_provider_prompts(self):
        with mock.patch('builtins.input', side_effect=['no','same','y','70','n','y']), redirect_stdout(io.StringIO()), \
             mock.patch('artificium.setup.discover_provider_models',side_effect=AssertionError('must not probe')):
            updated = SetupWizard(self.paths).reconfigure(SetupOptions(scope='harness'),interactive=True)
        self.assertEqual((updated.vision,updated.offload_threshold_percent),('no',70))

    def test_interactive_setup_asks_harness_before_provider_and_keeps_defaults(self):
        self.paths.config.unlink()
        responses = iter(['','same','n','n','custom','http://localhost:8000','local','50000','','n','y'])
        prompts = []
        def answer(prompt):
            prompts.append(prompt)
            return next(responses)
        with mock.patch('builtins.input',side_effect=answer),redirect_stdout(io.StringIO()), mock.patch('artificium.setup.discover_provider_models',return_value=ModelDiscovery()):
            SetupWizard(self.paths).run(SetupOptions(),interactive=True)
        self.assertIn('Vision',prompts[0])
        self.assertIn('Working-memory',prompts[1])
        self.assertIn('offloading',prompts[2])
        config = ConfigStore(self.paths).load()
        self.assertFalse(config.mandatory_offload)
        self.assertEqual(config.offload_threshold_percent,80)

    def test_default_off_does_not_block_ordinary_tools_under_pressure(self):
        engine = ScriptedEngine(call('write_file',path='mind/space/normal.txt',content='normal'))
        agent = self.agent(engine)
        self.fill_context(agent)
        agent.run_turn()
        self.assertEqual((self.paths.space/'normal.txt').read_text(),'normal')
        self.assertNotIn('mandatory_offload_pending',agent.tools._control())

    def test_mandatory_mode_withholds_whole_mixed_batch(self):
        engine = ScriptedEngine(call('save_memory',path='test/withheld',content='A lesson',retrieve_when='testing') +
                                call('write_file',path='mind/space/blocked.txt',content='blocked'))
        agent = self.agent(engine,mandatory_offload=True)
        self.fill_context(agent)
        agent.run_turn()
        self.assertFalse((self.paths.space/'blocked.txt').exists())
        self.assertFalse((self.paths.memory/'test/withheld.txt').exists())
        self.assertTrue(agent.tools._control()['mandatory_offload_pending'])
        self.assertIn('MANDATORY WORKING-MEMORY OFFLOADING',json.dumps(engine.requests[0]))

    def test_mandatory_mode_preserves_learning_and_resumes_only_after_two_stage_offload(self):
        checkpoint = 'Continue the project. Verified findings are saved in test/learning. Next action: write the final artifact.'
        engine = ScriptedEngine(
            call('offload_working_memory'),
            call('save_memory',path='test/learning',content='Keep the verified lesson from the large source.',retrieve_when='Continuing this project'),
            call('offload_working_memory',path='context/project-continuation',checkpoint=checkpoint,
                 retrieve_when='Continue this project',reflection_complete=True),
            call('write_file',path='mind/space/resumed.txt',content='resumed'),
        )
        agent = self.agent(engine,mandatory_offload=True,max_life_loop_rounds=4,
                           context_window_tokens=60000,working_memory_tokens=50000)
        self.fill_context(agent)
        agent.run_turn()
        self.assertEqual((self.paths.space/'resumed.txt').read_text(),'resumed')
        self.assertTrue((self.paths.memory/'test/learning.txt').exists())
        self.assertTrue((self.paths.memory/'context/project-continuation.txt').exists())
        self.assertTrue(list(self.paths.context_archive.glob('*.jsonl')))
        self.assertNotIn('mandatory_offload_pending',agent.tools._control())
        self.assertIn('WORKING-MEMORY OFFLOAD REFLECTION',json.dumps(engine.requests[1]))

    def test_failed_offload_keeps_gate_across_restart_even_if_usage_falls(self):
        # A generous window keeps this focused on the pending-gate behavior
        # rather than the exact token budget of the (larger, post-restructure) prompts.
        agent = self.agent(ScriptedEngine(call('offload_working_memory',reflection_complete=True)),
                           mandatory_offload=True, context_window_tokens=100000,
                           offload_threshold_percent=30)
        self.fill_context(agent)
        agent.run_turn()
        self.assertTrue(agent.tools._control()['mandatory_offload_pending'])
        restarted = self.agent(ScriptedEngine(call('sleep',reflection_complete=True)),
                               mandatory_offload=True, context_window_tokens=100000,
                               offload_threshold_percent=30)
        self.assertTrue(restarted._mandatory_offload_required(0))
        restarted.run_turn()
        self.assertIsNone(restarted.tools.sleep_request)
        self.assertTrue(restarted.tools._control()['mandatory_offload_pending'])

    def test_threshold_boundary_and_operator_disable(self):
        agent = self.agent(mandatory_offload=True,offload_threshold_percent=75)
        self.assertFalse(agent._mandatory_offload_required(37499))
        self.assertTrue(agent._mandatory_offload_required(37500))
        resumed = self.agent(ScriptedEngine(call('write_file',path='mind/space/disabled.txt',content='ok')),mandatory_offload=False)
        resumed.run_turn()
        self.assertTrue((self.paths.space/'disabled.txt').exists())
        self.assertNotIn('mandatory_offload_pending',resumed.tools._control())

    def test_mandatory_phase_can_update_meta_memory_but_cannot_edit_self(self):
        engine = ScriptedEngine(call('write_file',path='mind/meta_memory.md',content='Updated memory routes',mode='overwrite'),
                                call('write_file',path='mind/self.txt',content='Unrelated Self change',mode='overwrite'))
        agent = self.agent(engine,mandatory_offload=True,max_life_loop_rounds=2)
        original_self = self.paths.self_file.read_bytes()
        self.fill_context(agent)
        agent.run_turn()
        self.assertEqual(self.paths.meta_memory.read_text(),'Updated memory routes')
        self.assertEqual(self.paths.self_file.read_bytes(),original_self)
