#!/usr/bin/env python3
"""Drive a real Chrome/Chromium window over the DevTools protocol. Standard library only.

    python3 mind/tools/browser.py COMMAND [ARGS]

The browser is an ordinary, visible window with a persistent profile (logins
and cookies survive restarts), started by this script and left running between
calls. Pages are read as a compact accessibility snapshot, not pixels: each
interactive element gets a short id, and later commands act on that id.

  start [--headless] [--url URL]   install a browser if none is found, launch it
  install                          only find/install a browser, print its path
  stop | status
  goto URL | back | reload
  snapshot [--max N]               title, url and interactive elements: [12] button "Sign in"
  text [--max-chars N]             visible page text (full text saved, path printed)
  click ID
  type ID TEXT [--clear] [--submit]    focus ID, type like a person (\\n = Enter)
  key KEYS                         e.g. Enter, Tab, Escape, Control+a
  scroll up|down [--pages N]
  tabs | tab N | newtab [URL] | closetab
  shot [--out PATH]                PNG screenshot (view it with load_images)
  eval JS                          evaluate JavaScript in the page, print the result

Ids belong to the snapshot that produced them: after the page changes, take a
new snapshot. Exit codes: 0 ok, 1 error (the message says what to do next).
Without a graphical session the script says so; use `start --headless` then.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BROWSER_NAMES = ("google-chrome-stable", "google-chrome", "chromium", "chromium-browser",
                 "chrome", "brave-browser", "microsoft-edge")
CHROME_FOR_TESTING = ("https://googlechromelabs.github.io/chrome-for-testing/"
                      "last-known-good-versions-with-downloads.json")


class BrowserError(Exception):
    """An error whose message is shown to the agent as is."""


def _state_dir() -> Path:
    tools = Path(__file__).resolve().parent
    if tools.name == "tools" and (tools.parent / "space").is_dir():
        base = tools.parent / "space" / "browser"
    else:
        base = Path(tempfile.gettempdir()) / f"artificium-browser-{os.getuid()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


STATE_DIR = _state_dir()
PROFILE_DIR = STATE_DIR / "profile"
STATE_FILE = STATE_DIR / "state.json"


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE_FILE)


# --- minimal WebSocket client (RFC 6455) -------------------------------------

def encode_frame(payload: bytes, opcode: int = 1) -> bytes:
    """One masked client frame, as the protocol requires."""
    head = bytes([0x80 | opcode])
    n = len(payload)
    if n < 126:
        head += bytes([0x80 | n])
    elif n < 65536:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    else:
        head += bytes([0x80 | 127]) + struct.pack(">Q", n)
    mask = os.urandom(4)
    return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))


def read_frame(reader) -> tuple[int, bool, bytes]:
    """Read one frame: (opcode, final, payload)."""
    first = reader.read(2)
    if len(first) < 2:
        raise BrowserError("The browser closed the connection. Run `start` again.")
    opcode, final = first[0] & 0x0F, bool(first[0] & 0x80)
    n = first[1] & 0x7F
    if n == 126:
        n = struct.unpack(">H", reader.read(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", reader.read(8))[0]
    mask = reader.read(4) if first[1] & 0x80 else None
    payload = reader.read(n)
    if mask:
        payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return opcode, final, payload


class Connection:
    """A DevTools session over one WebSocket: send a command, wait for its reply."""

    def __init__(self, ws_url: str, timeout: float = 30):
        parts = urllib.parse.urlsplit(ws_url)
        self.sock = socket.create_connection((parts.hostname, parts.port), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        self.sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {parts.hostname}:{parts.port}\r\n"
                           "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                           f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        self.reader = self.sock.makefile("rb")
        status = self.reader.readline()
        if b" 101 " not in status:
            raise BrowserError(f"The browser refused the DevTools connection ({status.strip().decode(errors='replace')}).")
        while self.reader.readline() not in (b"\r\n", b""):
            pass
        self.next_id = 0

    def close(self) -> None:
        try:
            self.sock.sendall(encode_frame(b"", 8))
        except OSError:
            pass
        self.sock.close()

    def _message(self) -> dict:
        data, started = b"", False
        while True:
            opcode, final, payload = read_frame(self.reader)
            if opcode == 9:
                self.sock.sendall(encode_frame(payload, 10))
            elif opcode == 8:
                raise BrowserError("The browser closed the connection. Run `start` again.")
            elif opcode in (1, 2, 0):
                data, started = data + payload, True
                if final:
                    return json.loads(data)
            elif not started:
                continue

    def call(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        want = self.next_id
        self.sock.sendall(encode_frame(json.dumps({"id": want, "method": method, "params": params or {}}).encode()))
        while True:
            message = self._message()
            if message.get("id") == want:
                if "error" in message:
                    raise BrowserError(f"{method} failed: {message['error'].get('message', message['error'])}")
                return message.get("result", {})

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                  "awaitPromise": True})
        if "exceptionDetails" in result:
            detail = result["exceptionDetails"]
            raise BrowserError("JavaScript error: " + str(detail.get("exception", {}).get("description") or detail.get("text")))
        return result.get("result", {}).get("value")


# --- finding, installing and starting the browser -------------------------------

def find_browser() -> str | None:
    override = os.environ.get("ARTIFICIUM_BROWSER")
    if override and Path(override).exists():
        return override
    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in sorted(STATE_DIR.glob("install/**/chrome")):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    return None


def _os_release_id() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("ID="):
                return line[3:].strip().strip('"')
    except OSError:
        pass
    return ""


def _can_sudo() -> bool:
    if os.geteuid() == 0:
        return True
    return bool(shutil.which("sudo")) and subprocess.run(
        ["sudo", "-n", "true"], capture_output=True).returncode == 0


def _package_install_commands() -> list[list[str]]:
    # Ubuntu's chromium package is a snap shim whose confinement breaks custom
    # profile directories, so it is left to the download fallback.
    os_id = _os_release_id()
    if shutil.which("apt-get") and os_id == "debian":
        return [["apt-get", "install", "-y", "chromium"]]
    if shutil.which("dnf"):
        return [["dnf", "install", "-y", "chromium"]]
    if shutil.which("pacman"):
        return [["pacman", "-S", "--noconfirm", "chromium"]]
    if shutil.which("zypper"):
        return [["zypper", "--non-interactive", "install", "chromium"]]
    return []


def _download_chrome_for_testing() -> str:
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise BrowserError("No prebuilt browser for this CPU. Install Chromium with the system package "
                           "manager, or set ARTIFICIUM_BROWSER to a Chrome/Chromium binary.")
    with urllib.request.urlopen(CHROME_FOR_TESTING, timeout=30) as response:
        meta = json.load(response)["channels"]["Stable"]
    url = next(d["url"] for d in meta["downloads"]["chrome"] if d["platform"] == "linux64")
    target = STATE_DIR / "install"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    archive = target / "chrome.zip"
    with urllib.request.urlopen(url, timeout=600) as response, open(archive, "wb") as out:
        shutil.copyfileobj(response, out)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(target)  # zipfile drops the executable bits
    archive.unlink()
    for path in target.rglob("*"):
        if path.is_file() and not path.suffix and path.stat().st_size > 100_000:
            path.chmod(0o755)
    found = find_browser()
    if not found:
        raise BrowserError("The download finished but no browser binary was found in it.")
    return found


def ensure_browser() -> str:
    found = find_browser()
    if found:
        return found
    commands = _package_install_commands()
    if commands and _can_sudo():
        prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
        for command in commands:
            print("No browser found; installing: " + " ".join(prefix + command), flush=True)
            subprocess.run(prefix + command, capture_output=True, text=True, timeout=1200)
        found = find_browser()
        if found:
            return found
    print("No browser found; downloading Chrome for Testing into the workspace...", flush=True)
    return _download_chrome_for_testing()


def discover_display(env: dict) -> dict:
    """Add display variables when the process was started outside the desktop session."""
    env = dict(env)
    if env.get("DISPLAY") or env.get("WAYLAND_DISPLAY"):
        return env
    runtime = env.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    for socket_path in sorted(Path(runtime).glob("wayland-[0-9]*")):
        if not socket_path.name.endswith(".lock"):
            env.update(WAYLAND_DISPLAY=socket_path.name, XDG_RUNTIME_DIR=runtime)
            return env
    for socket_path in sorted(Path("/tmp/.X11-unix").glob("X[0-9]*")):
        env["DISPLAY"] = ":" + socket_path.name[1:]
        authority = Path.home() / ".Xauthority"
        if "XAUTHORITY" not in env and authority.exists():
            env["XAUTHORITY"] = str(authority)
        return env
    return env


def devtools_port() -> int | None:
    try:
        return int((PROFILE_DIR / "DevToolsActivePort").read_text().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def http_json(path: str, port: int, timeout: float = 5):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.load(response)


def running_port() -> int | None:
    port = devtools_port()
    if port is None:
        return None
    try:
        http_json("/json/version", port, timeout=2)
        return port
    except (OSError, ValueError):
        return None


def start(headless: bool, url: str | None) -> str:
    port = running_port()
    if port:
        return f"Browser already running (port {port}). Use `tabs` or `goto`."
    binary = ensure_browser()
    env = discover_display(os.environ)
    if not headless and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        raise BrowserError("No graphical session found (DISPLAY and WAYLAND_DISPLAY are unset). Start Artificium "
                           "from a terminal inside the desktop session, or use `start --headless`.")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    (PROFILE_DIR / "DevToolsActivePort").unlink(missing_ok=True)
    command = [binary, f"--user-data-dir={PROFILE_DIR}", "--remote-debugging-port=0",
               "--no-first-run", "--no-default-browser-check", "--ozone-platform=auto"]
    if headless:
        command.append("--headless=new")
    if os.geteuid() == 0:
        command.append("--no-sandbox")  # Chrome refuses to start as root otherwise
    command.append(url or "about:blank")
    with open(STATE_DIR / "browser.log", "ab") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=log, start_new_session=True)
    for _ in range(100):
        time.sleep(0.2)
        if running_port():
            save_state({"pid": process.pid, "binary": binary, "headless": headless})
            return f"Browser started ({'headless' if headless else 'visible window'}): {binary}"
        if process.poll() is not None:
            break
    raise BrowserError(f"The browser did not start; see {STATE_DIR / 'browser.log'}. Missing system "
                       "libraries are the usual cause on a minimal server.")


def stop() -> str:
    port = running_port()
    state = load_state()
    if port:
        try:
            connection = Connection(http_json("/json/version", port)["webSocketDebuggerUrl"])
            connection.call("Browser.close")
            connection.close()
        except (BrowserError, OSError):
            pass
    elif state.get("pid"):
        try:
            os.kill(state["pid"], 15)
        except OSError:
            pass
    save_state({})
    return "Browser stopped."


# --- tabs and pages ---------------------------------------------------------------

def pages(port: int) -> list[dict]:
    return [t for t in http_json("/json/list", port) if t.get("type") == "page"]


def current_page(port: int) -> dict:
    listed = pages(port)
    if not listed:
        raise BrowserError("No open tab. Use `newtab URL`.")
    wanted = load_state().get("target")
    return next((t for t in listed if t["id"] == wanted), listed[0])


def attach() -> tuple[Connection, dict, dict]:
    port = running_port()
    if not port:
        raise BrowserError("The browser is not running. Run `start` first.")
    page = current_page(port)
    return Connection(page["webSocketDebuggerUrl"]), page, load_state()


def wait_ready(connection: Connection, timeout: float = 20) -> None:
    deadline = time.time() + timeout
    time.sleep(0.3)
    while time.time() < deadline:
        try:
            if connection.evaluate("document.readyState") == "complete":
                return
        except BrowserError:
            pass  # the page is mid-navigation
        time.sleep(0.3)


def page_line(connection: Connection) -> str:
    wait_ready(connection)
    title = connection.evaluate("document.title")
    return f"Page: {title!r} - {connection.evaluate('location.href')}"


# --- snapshot -------------------------------------------------------------------------

INTERACTIVE = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem",
               "menuitemcheckbox", "menuitemradio", "tab", "switch", "slider", "spinbutton",
               "option", "listbox", "treeitem"}


def shorten(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def reduce_tree(nodes: list[dict], limit: int) -> tuple[list[str], dict[str, int]]:
    """Turn raw accessibility nodes into numbered lines and an id -> backend node map."""
    lines, ids = [], {}
    for node in nodes:
        if node.get("ignored"):
            continue
        role = (node.get("role") or {}).get("value", "")
        name = (node.get("name") or {}).get("value", "")
        if role == "heading" and name:
            lines.append(f"    # {shorten(name, 100)}")
        elif role in INTERACTIVE and node.get("backendDOMNodeId"):
            if len(ids) >= limit:
                continue
            key = str(len(ids) + 1)
            ids[key] = node["backendDOMNodeId"]
            props = {p["name"]: p.get("value", {}).get("value") for p in node.get("properties", [])}
            extra = ""
            value = (node.get("value") or {}).get("value")
            if value not in (None, ""):
                extra += f" value={shorten(value, 40)!r}"
            if props.get("checked") in (True, "true"):
                extra += " checked"
            if props.get("disabled"):
                extra += " disabled"
            lines.append(f"[{key}] {role} {shorten(name)!r}{extra}")
    return lines, ids


def snapshot(limit: int) -> str:
    connection, page, state = attach()
    try:
        header = page_line(connection)
        nodes = connection.call("Accessibility.getFullAXTree").get("nodes", [])
        lines, ids = reduce_tree(nodes, limit)
        state.update(target=page["id"], elements=ids)
        save_state(state)
        total = sum(1 for n in nodes if (n.get("role") or {}).get("value") in INTERACTIVE and not n.get("ignored"))
        more = f"\n(showing {len(ids)} of {total} interactive elements; raise --max or scroll)" if total > len(ids) else ""
        return header + "\n" + "\n".join(lines) + more
    finally:
        connection.close()


# --- input --------------------------------------------------------------------------------

KEYS = {"Enter": ("Enter", "Enter", 13, "\r"), "Tab": ("Tab", "Tab", 9, ""),
        "Escape": ("Escape", "Escape", 27, ""), "Backspace": ("Backspace", "Backspace", 8, ""),
        "Delete": ("Delete", "Delete", 46, ""), "ArrowUp": ("ArrowUp", "ArrowUp", 38, ""),
        "ArrowDown": ("ArrowDown", "ArrowDown", 40, ""), "ArrowLeft": ("ArrowLeft", "ArrowLeft", 37, ""),
        "ArrowRight": ("ArrowRight", "ArrowRight", 39, ""), "Home": ("Home", "Home", 36, ""),
        "End": ("End", "End", 35, ""), "PageUp": ("PageUp", "PageUp", 33, ""),
        "PageDown": ("PageDown", "PageDown", 34, ""), "Space": (" ", "Space", 32, " ")}
MODIFIERS = {"Alt": 1, "Control": 2, "Ctrl": 2, "Meta": 4, "Shift": 8}


def pause(low: float, high: float) -> None:
    time.sleep(random.uniform(low, high))


def press(connection: Connection, spec: str) -> None:
    *mods, name = spec.split("+")
    bits = 0
    for mod in mods:
        if mod not in MODIFIERS:
            raise BrowserError(f"Unknown modifier {mod!r}. Use Control, Alt, Shift or Meta.")
        bits |= MODIFIERS[mod]
    if name in KEYS:
        key, code, vk, text = KEYS[name]
    elif len(name) == 1:
        key, code, vk, text = name, ("Key" + name.upper() if name.isalpha() else name), ord(name.upper()), name
    else:
        raise BrowserError(f"Unknown key {name!r}. Use a single character or one of: {', '.join(KEYS)}.")
    if bits & (1 | 2 | 4):
        text = ""  # shortcuts do not insert text
    common = {"modifiers": bits, "key": key, "code": code, "windowsVirtualKeyCode": vk}
    connection.call("Input.dispatchKeyEvent", {"type": "keyDown", "text": text, **common} if text
                    else {"type": "rawKeyDown", **common})
    pause(0.03, 0.09)
    connection.call("Input.dispatchKeyEvent", {"type": "keyUp", **common})


def element_center(connection: Connection, state: dict, element: str) -> tuple[float, float]:
    backend = (state.get("elements") or {}).get(element)
    if backend is None:
        raise BrowserError(f"Unknown element id {element}. Take a new `snapshot` and use an id from it.")
    try:
        connection.call("DOM.enable")
        connection.call("DOM.getDocument", {"depth": 0})
        connection.call("DOM.scrollIntoViewIfNeeded", {"backendNodeId": backend})
        quad = connection.call("DOM.getContentQuads", {"backendNodeId": backend})["quads"][0]
    except (BrowserError, IndexError, KeyError):
        raise BrowserError(f"Element {element} is gone or not visible; the page changed. Take a new `snapshot`.")
    xs, ys = quad[0::2], quad[1::2]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    # Land somewhere inside the element rather than on the exact centre.
    return (min(xs) + width * random.uniform(0.35, 0.65), min(ys) + height * random.uniform(0.35, 0.65))


def move_and_click(connection: Connection, state: dict, x: float, y: float) -> None:
    last = state.get("mouse") or [x - random.uniform(80, 200), y - random.uniform(40, 120)]
    steps = random.randint(4, 8)
    for i in range(1, steps + 1):
        t = i / steps
        eased = t * t * (3 - 2 * t)
        connection.call("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": last[0] + (x - last[0]) * eased + random.uniform(-1.5, 1.5) * (1 - t),
            "y": last[1] + (y - last[1]) * eased + random.uniform(-1.5, 1.5) * (1 - t)})
        pause(0.008, 0.025)
    connection.call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
    pause(0.05, 0.15)
    for kind in ("mousePressed", "mouseReleased"):
        connection.call("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left",
                                                      "buttons": 1 if kind == "mousePressed" else 0,
                                                      "clickCount": 1})
        pause(0.04, 0.11)
    state["mouse"] = [x, y]


def describe_element(connection: Connection, state: dict, element: str) -> str:
    try:
        node = connection.call("DOM.describeNode", {"backendNodeId": state["elements"][element]})["node"]
        return node.get("nodeName", "").lower()
    except (BrowserError, KeyError):
        return "element"


def act_click(element: str) -> str:
    connection, page, state = attach()
    try:
        x, y = element_center(connection, state, element)
        move_and_click(connection, state, x, y)
        save_state(state)
        pause(0.4, 0.8)
        return f"Clicked [{element}]. " + page_line(connection)
    finally:
        connection.close()


def act_type(element: str, text: str, clear: bool, submit: bool) -> str:
    connection, page, state = attach()
    try:
        x, y = element_center(connection, state, element)
        move_and_click(connection, state, x, y)
        if clear:
            press(connection, "Control+a")
            press(connection, "Backspace")
        for char in text:
            if char == "\n":
                press(connection, "Enter")
            else:
                connection.call("Input.dispatchKeyEvent", {"type": "keyDown", "text": char, "key": char})
                pause(0.02, 0.06)
                connection.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": char})
            pause(0.03, 0.14)
        if submit:
            pause(0.2, 0.5)
            press(connection, "Enter")
        save_state(state)
        pause(0.3, 0.7)
        return f"Typed {len(text)} characters into [{element}]. " + page_line(connection)
    finally:
        connection.close()


def act_key(spec: str) -> str:
    connection, page, state = attach()
    try:
        press(connection, spec)
        pause(0.3, 0.7)
        return f"Pressed {spec}. " + page_line(connection)
    finally:
        connection.close()


def act_scroll(direction: str, pages_count: float) -> str:
    connection, page, state = attach()
    try:
        width = connection.evaluate("innerWidth")
        height = connection.evaluate("innerHeight")
        remaining = height * 0.85 * pages_count * (1 if direction == "down" else -1)
        while abs(remaining) > 1:
            step = max(-120, min(120, remaining)) * random.uniform(0.9, 1.1)
            connection.call("Input.dispatchMouseEvent", {"type": "mouseWheel", "x": width / 2, "y": height / 2,
                                                          "deltaX": 0, "deltaY": step})
            remaining -= step
            pause(0.01, 0.04)
        pause(0.3, 0.6)
        return f"Scrolled {direction}. Take a new `snapshot` to see what is there now."
    finally:
        connection.close()


def navigate(url: str | None, action: str) -> str:
    connection, page, state = attach()
    try:
        if action == "goto":
            if "://" not in url and not url.startswith("about:"):
                url = "https://" + url
            connection.call("Page.navigate", {"url": url})
        elif action == "reload":
            connection.call("Page.reload")
        else:
            history = connection.call("Page.getNavigationHistory")
            if history["currentIndex"] == 0:
                raise BrowserError("No earlier page in this tab's history.")
            connection.call("Page.navigateToHistoryEntry",
                            {"entryId": history["entries"][history["currentIndex"] - 1]["id"]})
        state.update(elements={})
        save_state(state)
        return page_line(connection)
    finally:
        connection.close()


def page_text(max_chars: int) -> str:
    connection, page, state = attach()
    try:
        header = page_line(connection)
        text = connection.evaluate("document.body ? document.body.innerText : ''") or ""
        path = STATE_DIR / "pages" / f"{int(time.time())}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
        excerpt = text[:max_chars] + (f"\n... ({len(text)} characters in all)" if len(text) > max_chars else "")
        return f"{header}\nFull text: {path}\n\n{excerpt}"
    finally:
        connection.close()


def tabs_command(action: str, argument: str | None) -> str:
    port = running_port()
    if not port:
        raise BrowserError("The browser is not running. Run `start` first.")
    state = load_state()
    listed = pages(port)
    if action == "tabs":
        current = current_page(port)["id"] if listed else None
        return "\n".join(f"{'*' if t['id'] == current else ' '} {i + 1}. {shorten(t.get('title', ''), 60)!r} {t['url']}"
                         for i, t in enumerate(listed)) or "No open tab."
    if action == "tab":
        try:
            target = listed[int(argument) - 1]
        except (ValueError, IndexError):
            raise BrowserError("No such tab number. Run `tabs` to list them.")
        state.update(target=target["id"], elements={})
        save_state(state)
        urllib.request.urlopen(f"http://127.0.0.1:{port}/json/activate/{target['id']}", timeout=5).read()
        return f"Switched to tab {argument}: {target['url']}"
    if action == "newtab":
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/json/new?{urllib.parse.quote(argument or 'about:blank', safe=':/?&=%')}", method="PUT")
        with urllib.request.urlopen(request, timeout=10) as response:
            target = json.load(response)
        state.update(target=target["id"], elements={})
        save_state(state)
        connection = Connection(target["webSocketDebuggerUrl"])
        try:
            return "Opened a new tab. " + page_line(connection)
        finally:
            connection.close()
    current = current_page(port)
    urllib.request.urlopen(f"http://127.0.0.1:{port}/json/close/{current['id']}", timeout=5).read()
    state.update(target=None, elements={})
    save_state(state)
    return "Closed the tab."


def screenshot(out: str | None) -> str:
    connection, page, state = attach()
    try:
        data = connection.call("Page.captureScreenshot", {"format": "png"})["data"]
    finally:
        connection.close()
    path = Path(out) if out else STATE_DIR / "shots" / f"{int(time.time())}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(data))
    return f"Screenshot saved: {path} (view it with load_images)"


def evaluate(expression: str) -> str:
    connection, page, state = attach()
    try:
        value = connection.evaluate(expression)
        return json.dumps(value, ensure_ascii=False, indent=2) if not isinstance(value, str) else value
    finally:
        connection.close()


def status() -> str:
    port = running_port()
    if not port:
        return "Browser not running. Binary: " + (find_browser() or "none found (`start` installs one)")
    state = load_state()
    return (f"Browser running (port {port}, {'headless' if state.get('headless') else 'visible'}). "
            f"Tabs:\n{tabs_command('tabs', None)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drive a real Chrome/Chromium window.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("start"); p.add_argument("--headless", action="store_true"); p.add_argument("--url")
    for name in ("install", "stop", "status", "back", "reload", "tabs", "closetab"):
        sub.add_parser(name)
    p = sub.add_parser("goto"); p.add_argument("url")
    p = sub.add_parser("snapshot"); p.add_argument("--max", type=int, default=150)
    p = sub.add_parser("text"); p.add_argument("--max-chars", type=int, default=4000)
    p = sub.add_parser("click"); p.add_argument("id")
    p = sub.add_parser("type"); p.add_argument("id"); p.add_argument("text")
    p.add_argument("--clear", action="store_true"); p.add_argument("--submit", action="store_true")
    p = sub.add_parser("key"); p.add_argument("keys")
    p = sub.add_parser("scroll"); p.add_argument("direction", choices=["up", "down"])
    p.add_argument("--pages", type=float, default=1)
    p = sub.add_parser("tab"); p.add_argument("number")
    p = sub.add_parser("newtab"); p.add_argument("url", nargs="?")
    p = sub.add_parser("shot"); p.add_argument("--out")
    p = sub.add_parser("eval"); p.add_argument("js")
    return parser


def run(args: argparse.Namespace) -> str:
    command = args.command
    if command == "start":
        return start(args.headless, args.url)
    if command == "install":
        return "Browser: " + ensure_browser()
    if command == "stop":
        return stop()
    if command == "status":
        return status()
    if command in ("goto", "back", "reload"):
        return navigate(getattr(args, "url", None), command)
    if command == "snapshot":
        return snapshot(args.max)
    if command == "text":
        return page_text(args.max_chars)
    if command == "click":
        return act_click(args.id)
    if command == "type":
        return act_type(args.id, args.text.replace("\\n", "\n"), args.clear, args.submit)
    if command == "key":
        return act_key(args.keys)
    if command == "scroll":
        return act_scroll(args.direction, args.pages)
    if command in ("tabs", "tab", "newtab", "closetab"):
        return tabs_command(command, getattr(args, "number", None) or getattr(args, "url", None))
    if command == "shot":
        return screenshot(args.out)
    return evaluate(args.js)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(run(args))
        return 0
    except BrowserError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Error: {error}. If the browser died, run `start` again.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
