"""Coverage for ``seed/tools/browser.py``: the WebSocket framing, the
accessibility-tree reduction, display discovery, key parsing, and (when a
Chrome/Chromium is installed) a headless end-to-end run against a local page.
The script is loaded by file path, as it ships under ``app/seed/tools``."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_seed_browser_under_test", ROOT / "seed/tools/browser.py")
browser = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = browser
spec.loader.exec_module(browser)

PAGE = (b"<html><head><title>Demo</title></head><body><h1>Hello</h1><a href='/two'>Go two</a>"
        b"<input id=q aria-label=Query><button onclick=\"document.title='clicked'\">Send</button></body></html>")


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE if self.path == "/" else b"<title>Two</title><h1>Two</h1>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FramingTests(unittest.TestCase):
    def test_client_frames_are_masked_and_round_trip(self):
        for size in (5, 300, 70000):
            payload = os.urandom(size)
            frame = browser.encode_frame(payload)
            self.assertTrue(frame[1] & 0x80)
            opcode, final, decoded = browser.read_frame(io.BytesIO(frame))
            self.assertEqual((opcode, final, decoded), (1, True, payload))


class ReductionTests(unittest.TestCase):
    def test_interactive_nodes_get_ids_and_ignored_ones_do_not(self):
        nodes = [
            {"role": {"value": "heading"}, "name": {"value": "Title"}},
            {"role": {"value": "button"}, "name": {"value": "Go"}, "backendDOMNodeId": 7},
            {"role": {"value": "button"}, "name": {"value": "Hidden"}, "backendDOMNodeId": 8, "ignored": True},
            {"role": {"value": "textbox"}, "name": {"value": "Q"}, "value": {"value": "abc"}, "backendDOMNodeId": 9},
            {"role": {"value": "link"}, "name": {"value": "Over limit"}, "backendDOMNodeId": 10},
        ]
        lines, ids = browser.reduce_tree(nodes, limit=2)
        self.assertEqual(ids, {"1": 7, "2": 9})
        self.assertIn("[2] textbox 'Q' value='abc'", lines)
        self.assertNotIn("Hidden", "\n".join(lines))
        self.assertNotIn("Over limit", "\n".join(lines))


class DisplayTests(unittest.TestCase):
    def test_existing_display_is_kept(self):
        self.assertEqual(browser.discover_display({"DISPLAY": ":5"})["DISPLAY"], ":5")

    def test_no_display_found_means_headless_is_suggested(self):
        with mock.patch.object(browser, "running_port", return_value=None), \
             mock.patch.object(browser, "ensure_browser", return_value="/bin/true"), \
             mock.patch.object(browser, "discover_display", return_value={}):
            with self.assertRaisesRegex(browser.BrowserError, "--headless"):
                browser.start(False, None)

    def test_unknown_key_is_actionable(self):
        with self.assertRaisesRegex(browser.BrowserError, "Unknown key"):
            browser.press(mock.Mock(), "Banana")


@unittest.skipUnless(browser.find_browser(), "no Chrome/Chromium installed")
class LiveTests(unittest.TestCase):
    def test_headless_snapshot_click_type_and_stop(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}/"
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(browser, "STATE_DIR", Path(tmp)), \
             mock.patch.object(browser, "PROFILE_DIR", Path(tmp) / "profile"), \
             mock.patch.object(browser, "STATE_FILE", Path(tmp) / "state.json"):
            def call(*argv):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertEqual(browser.main(list(argv)), 0)
                return out.getvalue()
            try:
                call("start", "--headless", "--url", base)
                snap = call("snapshot")
                self.assertIn("Demo", snap)
                self.assertIn("link 'Go two'", snap)
                button = next(line for line in snap.splitlines() if "button 'Send'" in line)
                call("click", button[1:button.index("]")])
                self.assertEqual(call("eval", "document.title").strip(), "clicked")
                call("goto", base + "x")
                self.assertIn("Two", call("text"))
            finally:
                call("stop")
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
